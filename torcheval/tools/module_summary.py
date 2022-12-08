# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
import warnings
from collections import defaultdict, deque
from typing import Any, Callable, DefaultDict, Deque, Dict, List, Optional, Tuple, Union

import torch
from torch.nn.parameter import UninitializedParameter
from torch.utils._pytree import PyTree, tree_flatten
from torch.utils.hooks import RemovableHandle
from torcheval.tools.flops import flop_mapping, FlopTensorDispatchMode

from torchtnt.utils.version import is_torch_version_geq_1_13
from typing_extensions import Literal

_ATTRIB_TO_COL_HEADER = {
    "module_name": "Name",
    "module_type": "Type",
    "num_parameters": "# Parameters",
    "num_trainable_parameters": "# Trainable Parameters",
    "size_bytes": "Size (bytes)",
    "has_uninitialized_param": "Contains Uninitialized Parameters?",
    "flops_forward": "Forward FLOPs",
    "flops_backward": "Backward FLOPs",
    "in_size": "In size",
    "out_size": "Out size",
}  # Attribute: column header (in table)
_ATTRIBS: List[str] = list(_ATTRIB_TO_COL_HEADER.keys())
_FLOP_ATTRIBS: List[str] = ["flops_forward", "flops_backward"]


_PARAMETER_NUM_UNITS = [" ", "K", "M", "B", "T"]
_PARAMETER_FLOPS_UNITS = [" ", "k", "M", "G", "T", "P", "E", "Z", "Y"]

TUnknown = Literal["?"]
_UNKNOWN_SIZE: TUnknown = "?"


class ModuleSummary:
    """
    Summary of module and its submodules. It collects the following information:

    - Name
    - Type
    - Number of parameters
    - Number of trainable parameters
    - Estimated size in bytes
    - Contains uninitialized parameters
    - FLOPs for forward (-1 meaning not calculated)
    - FLOPs for backward (-1 meaning not calculated)
    - Input shape ("?" meaning not calculated)
    - Output shape ("?" meaning not calculated)
    """

    def __init__(self) -> None:
        self._module_name: str = ""
        self._module_type: str = ""
        self._num_parameters: int = 0
        self._num_trainable_parameters: int = 0
        self._size_bytes: int = 0
        self._submodule_summaries: Dict[str, "ModuleSummary"] = {}
        self._has_uninitialized_param: bool = False
        self._flops_forward: int = -1
        self._flops_backward: int = -1
        self._flops_forward_detail: Dict[str, int] = {}
        self._flops_backward_detail: Dict[str, int] = {}

        self._in_size: Union[TUnknown, List[int]] = _UNKNOWN_SIZE
        self._out_size: Union[TUnknown, List[int]] = _UNKNOWN_SIZE

    @property
    def submodule_summaries(self) -> Dict[str, "ModuleSummary"]:
        """
        A Dict with the names of submodules as keys and corresponding ModuleSummary
        objects as values. These can be traversed for visualization.
        """
        return self._submodule_summaries

    @property
    def module_name(self) -> str:
        """Returns the name of this module"""
        return self._module_name

    @property
    def module_type(self) -> str:
        """Returns the type of this module."""
        return self._module_type

    @property
    def num_parameters(self) -> int:
        """Returns the total number of parameters in this module."""
        if self.has_uninitialized_param:
            warnings.warn(
                "A layer with UninitializedParameter was found. "
                "Thus, the total number of parameters detected may be inaccurate."
            )
        return self._num_parameters

    @property
    def num_trainable_parameters(self) -> int:
        """
        Returns the total number of trainable parameters (requires_grad=True)
        in this module.
        """
        if self.has_uninitialized_param:
            warnings.warn(
                "A layer with UninitializedParameter was found. "
                "Thus, the total number of parameters detected may be inaccurate."
            )
        return self._num_trainable_parameters

    @property
    def flops_forward(self) -> int:
        """Returns the total FLOPs for forward calculation using this module."""
        if self.has_uninitialized_param:
            warnings.warn(
                "A layer with UninitializedParameter was found. "
                "Thus, the total number of FLOPs detected may be inaccurate."
            )
        return self._flops_forward

    @property
    def flops_backward(self) -> int:
        """Returns the total FLOPs for backward calculation using this module."""
        if self.has_uninitialized_param:
            warnings.warn(
                "A layer with UninitializedParameter was found. "
                "Thus, the total number of FLOPs detected may be inaccurate."
            )
        return self._flops_backward

    @property
    def in_size(self) -> Union[TUnknown, List[int]]:
        """Returns the input size of the module"""
        return self._in_size

    @property
    def out_size(self) -> Union[TUnknown, List[int]]:
        """Returns the output size of the module"""
        return self._out_size

    @property
    def size_bytes(self) -> int:
        """Returns the total estimated size in bytes of a module."""
        if self.has_uninitialized_param:
            warnings.warn(
                "A layer with UninitializedParameter was found. "
                "Thus, the total byte sizes detected may be inaccurate."
            )
        return self._size_bytes

    @property
    def has_uninitialized_param(self) -> bool:
        """Returns if a parameter in this module is uninitialized"""
        return self._has_uninitialized_param

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return get_summary_table(self)


def _clean_flops(flop: DefaultDict[str, DefaultDict[str, int]], N: int) -> None:
    for _, sub_flop in flop.items():
        for opr in sub_flop:
            sub_flop[opr] = sub_flop[opr] // N


def _get_module_flops_and_activation_sizes(
    module: torch.nn.Module,
    # pyre-ignore
    module_args: Optional[Tuple[Any, ...]] = None,
    module_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[
    DefaultDict[str, DefaultDict[str, int]],  # mapping to store forward flops
    DefaultDict[str, DefaultDict[str, int]],  # mapping to store backward flops
    Dict[
        str, Tuple[Union[TUnknown, List[int]], Union[TUnknown, List[int]]]
    ],  # mapping from module name to activation size tuple (in_size, out_size)
]:
    if not is_torch_version_geq_1_13():
        raise RuntimeError(
            "Computing FLOPS requires PyTorch 1.13 or higher. "
            "Please install PyTorch 1.13 or higher to continue: https://pytorch.org/get-started/locally/"
        )
    # a mapping from module name to activation size tuple (in_size, out_size)
    activation_sizes: Dict[
        str, Tuple[Union[TUnknown, List[int]], Union[TUnknown, List[int]]]
    ] = {}
    handles: List[RemovableHandle] = []

    # pyre-ignore: Missing return annotation [3]
    def activation_size_hook(
        module_name: str,
    ) -> Callable[[torch.nn.Module, Any, Any], None]:
        # pyre-ignore: Missing parameter annotation [2]
        def hook(_: torch.nn.Module, inp: Any, out: Any) -> None:
            if len(inp) == 1:
                inp = inp[0]
            in_size = _parse_batch_shape(inp)
            out_size = _parse_batch_shape(out)
            activation_sizes[module_name] = (in_size, out_size)

        return hook

    # place forward hooks on all modules + submodules
    if not isinstance(module, torch.jit.ScriptModule):
        queue: Deque[Tuple[str, torch.nn.Module]] = deque([("", module)])
        while len(queue) > 0:
            module_name, mod = queue.pop()
            hook = activation_size_hook(module_name)
            handle = mod.register_forward_hook(hook)
            handles.append(handle)

            for name, submodule in mod.named_children():
                formatted_name = f"{module_name}.{name}" if module_name != "" else name
                queue.append((formatted_name, submodule))
    else:
        warnings.warn("Registering hooks on torch.jit.ScriptModule is not supported.")

    # TODO: here we assume module's output should be a single tensor.
    #   Need to provide more flexible implementation that support other types of output.
    module.zero_grad()
    with FlopTensorDispatchMode(module) as ftdm:
        # Count for forward flops (+ compute activation sizes)
        module_args = module_args or ()
        module_kwargs = module_kwargs or {}
        res = module(*module_args, **module_kwargs)

        # detach activation size hook handles
        for hook_handle in handles:
            hook_handle.remove()

        flops_forward = copy.deepcopy(ftdm.flop_counts)
        flops_backward = None
        if isinstance(res, torch.Tensor):
            # Count for backward flops
            ftdm.reset()
            res.mean().backward()
            flops_backward = copy.deepcopy(ftdm.flop_counts)

    # Reverting all the changes:
    module.zero_grad()
    # TODO: Reverting BN: We also need to save status of BN running mean/var before running and revert those.
    return flops_forward, flops_backward, activation_sizes


def _has_uninitialized_param(module: torch.nn.Module) -> bool:
    for param in module.parameters():
        if isinstance(param, UninitializedParameter):
            return True
    return False


def _has_tensor(item: Optional[PyTree]) -> bool:
    flattened_list, _ = tree_flatten(item)
    for ele in flattened_list:
        if isinstance(ele, torch.Tensor):
            return True
    return False


def get_module_summary(
    module: torch.nn.Module,
    # pyre-ignore
    module_args: Optional[Tuple[Any, ...]] = None,
    module_kwargs: Optional[Dict[str, Any]] = None,
) -> ModuleSummary:
    """
    Generate a :class:`~ModuleSummary` object, then assign its values and generate submodule tree.

    Args:
        module: The module to be summarized.
        module_args: A tuple of arguments for the module to run and calculate FLOPs and activation sizes.
        module_kwargs: Any kwarg arguments to be passed into the module's forward function.

            Note:
              If passing module_args/module_kwargs for FLOPs and activation sizes, must be using PyTorch 1.13+.

            Note:
              If module contains any lazy submodule, we will NOT calculate FLOPs.

            Note:
              Currently only modules that output a single tensor are supported.
              TODO: to support more flexible output for module.

    """

    flops_forward = None
    flops_backward = None
    activation_sizes: Dict[
        str, Tuple[Union[TUnknown, List[int]], Union[TUnknown, List[int]]]
    ] = {}
    has_uninitialized_param = _has_uninitialized_param(module)
    if not has_uninitialized_param:
        has_tensor_in_args = _has_tensor(module_args)
        has_tensor_in_kwargs = _has_tensor(module_kwargs)
        if has_tensor_in_kwargs:
            warnings.warn(
                "A tensor in kwargs was found. This may lead to an inaccurately computed activation size, as keyword arguments are not passed into forward hooks for modules. "
                "For best results, please input tensors though args."
            )
        if has_tensor_in_args or has_tensor_in_kwargs:
            (
                flops_forward,
                flops_backward,
                activation_sizes,
            ) = _get_module_flops_and_activation_sizes(
                module, module_args, module_kwargs
            )

    return _generate_module_summary(
        module,
        "",
        flops_forward=flops_forward,
        flops_backward=flops_backward,
        activation_sizes=activation_sizes,
    )


def _generate_module_summary(
    module: torch.nn.Module,
    module_name: str,
    activation_sizes: Dict[
        str, Tuple[Union[TUnknown, List[int]], Union[TUnknown, List[int]]]
    ],
    flops_forward: Optional[DefaultDict[str, DefaultDict[str, int]]] = None,
    flops_backward: Optional[DefaultDict[str, DefaultDict[str, int]]] = None,
) -> ModuleSummary:
    """
    Recursively generate and populate metrics for ModelSummary.
    """
    module_summary = ModuleSummary()
    module_summary._module_name = module_name
    module_summary._module_type = str(module.__class__.__name__)

    for name, submodule in module.named_children():

        formatted_name = f"{module_name}.{name}" if module_name != "" else name

        submodule_summary = _generate_module_summary(
            submodule,
            formatted_name,
            flops_forward=flops_forward,
            flops_backward=flops_backward,
            activation_sizes=activation_sizes,
        )

        # Add results from submodule summary
        module_summary._submodule_summaries[formatted_name] = submodule_summary
        module_summary._has_uninitialized_param = (
            module_summary._has_uninitialized_param
            or submodule_summary._has_uninitialized_param
        )
        module_summary._num_parameters += submodule_summary._num_parameters
        module_summary._num_trainable_parameters += (
            submodule_summary._num_trainable_parameters
        )
        module_summary._size_bytes += submodule_summary._size_bytes

    for param in module.parameters(recurse=False):
        if isinstance(param, UninitializedParameter):
            module_summary._has_uninitialized_param = True
        else:
            numel = param.numel()
            module_summary._num_parameters += numel
            module_summary._size_bytes += numel * param.element_size()
            if param.requires_grad:
                module_summary._num_trainable_parameters += numel

    for buf in module.buffers(recurse=False):
        module_summary._size_bytes += buf.numel() * buf.element_size()

    # Calculate flops forward/backward.
    if flops_forward is not None:
        module_summary._flops_forward_detail = dict(flops_forward[module_name])
        module_summary._flops_forward = sum(
            [v for k, v in flops_forward[module_name].items()]
        )
    if flops_backward is not None:
        module_summary._flops_backward_detail = dict(flops_backward[module_name])
        module_summary._flops_backward = sum(
            [v for k, v in flops_backward[module_name].items()]
        )

    # set activation sizes
    if module_name in activation_sizes:
        in_size, out_size = activation_sizes[module_name]
        module_summary._in_size = in_size
        module_summary._out_size = out_size

    return module_summary


def get_summary_table(
    module_summary: ModuleSummary, human_readable_nums: bool = True
) -> str:
    """
    Generates a string summary_table, tabularizing the information in module_summary.

    Args:
        module_summary: module_summary to be printed/tabularized
        human_readable_nums: set to False for exact (e.g. 1234 vs 1.2 K)
    """
    stop_attr: List[str] = []
    # Unpack attributes
    if module_summary.flops_forward == -1:
        stop_attr.append("flops_forward")
    if module_summary.flops_backward == -1:
        stop_attr.append("flops_backward")
    if module_summary.in_size == _UNKNOWN_SIZE:
        stop_attr.append("in_size")
    if module_summary.out_size == _UNKNOWN_SIZE:
        stop_attr.append("out_size")
    unpacked_attribs, col_widths = defaultdict(list), defaultdict(int)
    _unpack_attributes(
        {"root": module_summary},
        unpacked_attribs,
        col_widths,
        human_readable_nums,
        stop_attr,
    )

    # Generate formatted summary_table string
    s = "{:{}}"  # inner {}: col_width
    use_attribs = [attr for attr in _ATTRIBS if attr not in stop_attr]
    n_rows, n_cols = len(unpacked_attribs[use_attribs[0]]), len(use_attribs)
    total_width = sum(col_widths.values()) + 3 * (n_cols - 1)

    header = [
        s.format(col_header, col_width)
        for col_header, col_width in zip(
            _ATTRIB_TO_COL_HEADER.values(), col_widths.values()
        )
    ]
    summary_table = " | ".join(header) + "\n" + "-" * total_width + "\n"

    for i in range(n_rows):
        row = []
        for attrib in use_attribs:
            row.append(unpacked_attribs[attrib][i])
            row = [
                s.format(r, col_width) for r, col_width in zip(row, col_widths.values())
            ]
        summary_table += " | ".join(row) + "\n"
    # Add disclaims for FLOPs:
    if "flops_forward" not in stop_attr or "flops_backward" not in stop_attr:
        used_operators = "|".join(
            [
                f"`{j.__name__}`"
                for j in flop_mapping.keys()
                if not j.__name__.endswith(".default")
            ]
        )
        summary_table += (
            f"Remark for FLOPs calculation: (1) Only operators {used_operators} are included. "
            + "To add more operators supported in FLOPs calculation, "
            + "please contribute to torcheval/tools/flops.py. "
            + "(2) The calculation related to additional loss function is not included. "
            + "For forward, we calculated FLOPs based on `loss = model(input_data).mean()`. "
            + "For backward, we calculated FLOPs based on `loss.backward()`. \n"
        )
    return summary_table


def prune_module_summary(module_summary: ModuleSummary, *, max_depth: int) -> None:
    """
    Prune the module summaries that are deeper than max_depth in the module
    summary tree. The ModuleSummary object is prunned inplace.

    Args:
        module_summary: Root module summary to prune.
        max_depth: The maximum depth of module summaries to keep.

    Raises:
        ValueError:
            If `max_depth` is an int less than 1
    """
    if max_depth < 1:
        raise ValueError(f"`max_depth` must be an int greater than 0. Got {max_depth}.")
    if max_depth == 1:
        module_summary._submodule_summaries = {}
        return

    for submodule_summary in module_summary._submodule_summaries.values():
        prune_module_summary(submodule_summary, max_depth=max_depth - 1)


def _unpack_attributes(
    module_summaries: Dict[str, ModuleSummary],
    unpacked_attribs: Dict[str, List[str]],
    col_widths: Dict[str, int],
    human_readable_nums: bool = True,
    stop_attr: Optional[List[str]] = None,
) -> None:
    """
    Unpack/flatten recursive module_summaries into table columns and store in unpacked_attribs.
    Also, populate col_widths (with max column width).

    Args:
        module_summaries: dict of module summaries
        unpacked_attribs: collects unpacked/flattened columns
        col_widths: tracks max table width for each column
        human_readable_nums: human readable nums (e.g. 1.2 K for 1234)
        stop_attr: a list of attributes that we stop from adding to the table,
        i.e. exclude from _ATTRIBS
    """

    if not module_summaries:
        return

    for module_summary in module_summaries.values():
        for attrib in _ATTRIBS:
            if stop_attr is not None and attrib in stop_attr:
                continue

            # Convert attribute value to string appropriately
            attrib_val = getattr(module_summary, attrib)
            if isinstance(attrib_val, bool):
                attrib_val = "Yes" if attrib_val else "No"
            elif isinstance(attrib_val, int) and attrib in _FLOP_ATTRIBS:
                if attrib_val < 0:
                    attrib_val = ""
                else:
                    attrib_val = (
                        _get_human_readable_count(
                            attrib_val, labels=_PARAMETER_FLOPS_UNITS
                        )
                        if human_readable_nums
                        else str(attrib_val)
                    )
            elif isinstance(attrib_val, int):
                attrib_val = (
                    _get_human_readable_count(attrib_val)
                    if human_readable_nums
                    else str(attrib_val)
                )
            elif isinstance(attrib_val, list):
                attrib_val = str(attrib_val)
            elif attrib_val is None:
                attrib_val = ""

            # Store converted attribute value, track max column width
            unpacked_attribs[attrib].append(attrib_val)
            col_widths[attrib] = max(
                len(_ATTRIB_TO_COL_HEADER[attrib]),
                len(attrib_val),
                col_widths[attrib],
            )
        # Recurse
        _unpack_attributes(
            module_summary.submodule_summaries,
            unpacked_attribs,
            col_widths,
            human_readable_nums,
            stop_attr,
        )


def _get_human_readable_count(number: int, labels: Optional[List[str]] = None) -> str:
    """Abbreviates an integer number with K, M, B, T for thousands, millions, billions and trillions, respectively.
    Examples:
        >>> _get_human_readable_count(123)
        '123  '
        >>> _get_human_readable_count(1234)  # (one thousand)
        '1.2 K'
        >>> _get_human_readable_count(int(2e6))   # (two million)
        '2.0 M'
        >>> _get_human_readable_count(int(3e9))   # (three billion)
        '3.0 B'
        >>> _get_human_readable_count(int(3e9), labels=[" ", "K", "M", "G", "T"])  # (Using units for FLOPs, 3 G)
        '3.0 G'
        >>> _get_human_readable_count(int(4e14))  # (four hundred trillion)
        '400 T'
        >>> _get_human_readable_count(int(5e15))  # (more than trillion)
        '5,000 T'
    Args:
        number: a positive integer number
        labels: a list of units that we want to use per 10^3 digits. Defaults to [" ", "K", "M", "B", "T"]
    Return:
        A string formatted according to the pattern described above.
    Raises:
        ValueError:
            If `number` is less than 0
        TypeError:
            If `number` is not an int
    """
    # logic does not work for floats (e.g. number=0.5)
    if not isinstance(number, int):
        raise TypeError(f"Input type must be int, but received {type(number)}")
    if number < 0:
        raise ValueError(f"Input value must be greater than 0, received {number}")
    labels = labels or _PARAMETER_NUM_UNITS
    if len(labels) <= 0:
        raise ValueError(
            f"Input labels must be a list with at least one string, received {labels}"
        )

    num_digits = int(math.floor(math.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(math.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))  # don't abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    number = number * (10**shift)
    index = num_groups - 1
    if index < 1 or number >= 100:
        return f"{int(number):,d} {labels[index]}"

    return f"{number:,.1f} {labels[index]}"


def _parse_batch_shape(batch: torch.Tensor) -> Union[TUnknown, List[int]]:
    if hasattr(batch, "shape"):
        return list(batch.shape)

    if isinstance(batch, (list, tuple)):
        shape = [_parse_batch_shape(el) for el in batch]
        return shape

    return _UNKNOWN_SIZE
