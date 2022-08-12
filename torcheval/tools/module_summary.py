# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
import warnings
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple

import torch
from torch.nn.parameter import UninitializedParameter

from torcheval.tools.flops import (
    flop_mapping,
    FlopTensor,
    instrument_module,
    start_counting,
)

_ATTRIB_TO_COL_HEADER = {
    "module_name": "Name",
    "module_type": "Type",
    "num_parameters": "# Parameters",
    "num_trainable_parameters": "# Trainable Parameters",
    "size_bytes": "Size (bytes)",
    "has_uninitialized_param": "Contains Uninitialized Parameter?",
    "flops_forward": "Forward FLOPs",
    "flops_backward": "Backward FLOPs",
}  # Attribute: column header (in table)
_ATTRIBS: List[str] = list(_ATTRIB_TO_COL_HEADER.keys())
_FLOP_ATTRIBS: List[str] = ["flops_forward", "flops_backward"]


_PARAMETER_NUM_UNITS = [" ", "K", "M", "B", "T"]
_PARAMETER_FLOPS_UNITS = [" ", "k", "M", "G", "T", "P", "E", "Z", "Y"]


class ModuleSummary:
    """
    Summary of module and its submodules. It collects the following information:
    - Name
    - Type
    - Number of parameters
    - Number of trainable parameters
    - Estimated size in bytes
    - Contains uninitialized parameter
    - FLOPs for forward (-1 meaning not calculated)
    - FLOPs for backward (-1 meaning not calculated)
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


def _get_module_flops(
    module: torch.nn.Module, module_input: torch.Tensor
) -> Tuple[
    DefaultDict[str, DefaultDict[str, int]], DefaultDict[str, DefaultDict[str, int]]
]:
    # TODO: here we assume module_input should be a single tensor and module's output should be a single tensor.
    #   Need to provide more flexible implementation that support other type of input/output.
    all_hooks: List[torch.utils.hooks.RemovableHandle] = []
    instrument_module(module, all_hooks, "")
    module_input = FlopTensor(module_input)
    N = len(module_input)
    module.zero_grad()

    # Count for forward flops
    start_counting()
    res = module(module_input).mean()
    flops_forward = copy.deepcopy(FlopTensor.flop_counts)
    # Count for backward flops
    start_counting()
    res.backward()
    flops_backward = copy.deepcopy(FlopTensor.flop_counts)

    # Norm FLOPs if N>1
    if N > 1:
        _clean_flops(flops_forward, N)
        _clean_flops(flops_backward, N)

    # Reverting all the changes:
    module.zero_grad()
    # TODO: Reverting BN: We also need to save status of BN running mean/var before running and revert those.
    for hood_handle in all_hooks:
        hood_handle.remove()
    return flops_forward, flops_backward


def _has_uninitialized_param(module: torch.nn.Module) -> bool:
    for param in module.parameters():
        if isinstance(param, UninitializedParameter):
            return True
    return False


def get_module_summary(
    module: torch.nn.Module,
    module_input: Optional[torch.Tensor] = None,
) -> ModuleSummary:
    """
    Generate a ModuleSummary object, then assign its values and generate submodule tree.

    Args:
        module: The module to be summarized.
        Value must be greater than 0. Set to `None` to include
        all submodules.
        module_name: The name of the current module.
        module_input: An input for the module to run and calculate FLOPs.
        Note: if module contains any lazy submodule, we will NOT calculate FLOPs.
        It should only contain 1 sample (i.e. batch_size=1), otherwise, we will return FLOPs = total FLOPs / len(module_input).
        Note: currently we only support module that takes a single tensor (i.e. module_input should be a torch.Tensor)
            and outputs a single tensor. TODO: to support more flexible input and output for module.


    """

    flops_forward = None
    flops_backward = None
    has_uninitialized_param = _has_uninitialized_param(module)
    if module_input is not None and not has_uninitialized_param:
        flops_forward, flops_backward = _get_module_flops(module, module_input)
    return _generate_module_summary(
        module,
        "",
        flops_forward=flops_forward,
        flops_backward=flops_backward,
    )


def _generate_module_summary(
    module: torch.nn.Module,
    module_name: str,
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
