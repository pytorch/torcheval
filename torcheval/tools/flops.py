# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import operator
from collections import defaultdict
from functools import reduce
from numbers import Number
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Tuple

import torch
from torch.utils._pytree import PyTree, tree_map

aten: torch._ops._OpNamespace = torch.ops.aten


# pyre-ignore [2] we don't care the type in outputs
def _matmul_flop_jit(inputs: Tuple[torch.Tensor], outputs: Tuple[Any]) -> Number:
    """
    Count flops for matmul.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two matrices.
    input_shapes = [v.shape for v in inputs]
    assert len(input_shapes) == 2, input_shapes
    assert input_shapes[0][-1] == input_shapes[1][-2], input_shapes
    flop = inputs[0].numel() * input_shapes[-1][-1]
    return flop


# pyre-ignore [2] we don't care the type in outputs
def _addmm_flop_jit(inputs: Tuple[torch.Tensor], outputs: Tuple[Any]) -> Number:
    """
    Count flops for fully connected layers.
    """
    # Count flop for nn.Linear
    # inputs is a list of length 3.
    input_shapes = [v.shape for v in inputs[1:3]]
    # input_shapes[0]: [batch size, input feature dimension]
    # input_shapes[1]: [batch size, output feature dimension]
    assert len(input_shapes[0]) == 2, input_shapes[0]
    assert len(input_shapes[1]) == 2, input_shapes[1]
    batch_size, input_dim = input_shapes[0]
    output_dim = input_shapes[1][1]
    flops = batch_size * input_dim * output_dim
    return flops


# pyre-ignore [2] we don't care the type in outputs
def _bmm_flop_jit(inputs: Tuple[torch.Tensor], outputs: Tuple[Any]) -> Number:
    """
    Count flops for the bmm operation.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two tensor.
    assert len(inputs) == 2, len(inputs)
    input_shapes = [v.shape for v in inputs]
    n, c, t = input_shapes[0]
    d = input_shapes[-1][-1]
    flop = n * c * t * d
    return flop


def _conv_flop_count(
    x_shape: List[int],
    w_shape: List[int],
    out_shape: List[int],
    transposed: bool = False,
) -> Number:
    """
    Count flops for convolution. Note only multiplication is
    counted. Computation for addition and bias is ignored.
    Flops for a transposed convolution are calculated as
    flops = (x_shape[2:] * prod(w_shape) * batch_size).
    Args:
        x_shape (list(int)): The input shape before convolution.
        w_shape (list(int)): The filter shape.
        out_shape (list(int)): The output shape after convolution.
        transposed (bool): is the convolution transposed
    Returns:
        int: the number of flops
    """
    batch_size = x_shape[0]
    conv_shape = (x_shape if transposed else out_shape)[2:]
    flop = (
        batch_size
        * reduce(operator.mul, w_shape, 1)
        * reduce(operator.mul, conv_shape, 1)
    )
    return flop


def _conv_flop_jit(
    inputs: Tuple[Any],  # pyre-ignore [2] the inputs can be union of Tensor/bool/Tuple
    outputs: Tuple[torch.Tensor],
) -> Number:
    """
    Count flops for convolution.
    """
    x: torch.Tensor = inputs[0]
    w: torch.Tensor = inputs[1]
    x_shape, w_shape, out_shape = (x.shape, w.shape, outputs[0].shape)
    transposed: bool = inputs[6]

    return _conv_flop_count(
        list(x_shape), list(w_shape), list(out_shape), transposed=transposed
    )


def _transpose_shape(shape: torch.Size) -> List[int]:
    return [shape[1], shape[0]] + list(shape[2:])


# pyre-ignore [2] the inputs can be union of Tensor/bool/Tuple & we don't care about outputs
def _conv_backward_flop_jit(inputs: Tuple[Any], outputs: Tuple[Any]) -> Number:
    grad_out_shape, x_shape, w_shape = [i.shape for i in inputs[:3]]
    output_mask = inputs[-1]
    fwd_transposed = inputs[7]
    flop_count: Number = 0

    if output_mask[0]:
        grad_input_shape = outputs[0].shape
        # pyre-ignore [58] this is actually sum of Number and Number
        flop_count = flop_count + _conv_flop_count(
            grad_out_shape, w_shape, grad_input_shape, not fwd_transposed
        )
    if output_mask[1]:
        grad_weight_shape = outputs[1].shape
        flop_count += _conv_flop_count(
            list(_transpose_shape(x_shape)),
            list(grad_out_shape),
            list(grad_weight_shape),
            fwd_transposed,
        )

    return flop_count


# pyre-ignore [5]
flop_mapping: Dict[Callable[..., Any], Callable[[Tuple[Any], Tuple[Any]], Number]] = {
    aten.mm: _matmul_flop_jit,
    aten.matmul: _matmul_flop_jit,
    aten.addmm: _addmm_flop_jit,
    aten.bmm: _bmm_flop_jit,
    aten.convolution: _conv_flop_jit,
    aten._convolution: _conv_flop_jit,
    aten.convolution_backward: _conv_backward_flop_jit,
    # Add their default to make sure they can be mapped.
    aten.mm.default: _matmul_flop_jit,
    aten.matmul.default: _matmul_flop_jit,
    aten.addmm.default: _addmm_flop_jit,
    aten.bmm.default: _bmm_flop_jit,
    aten.convolution.default: _conv_flop_jit,
    aten._convolution.default: _conv_flop_jit,
    aten.convolution_backward.default: _conv_backward_flop_jit,
}


# pyre-ignore [2, 3] it can be Tuple of anything.
def _normalize_tuple(x: Any) -> Tuple[Any]:
    if not isinstance(x, tuple):
        return (x,)
    return x


# pyre-fixme [13] elem is in __slots__ but obtain an error if initialize it.
class FlopTensor(torch.Tensor):
    elem: torch.Tensor
    flop_counts: DefaultDict[str, DefaultDict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    parents: List[str] = [""]

    __slots__ = ["elem"]

    @staticmethod
    def __new__(cls, elem: torch.Tensor) -> torch.Tensor:
        # The wrapping tensor (FlopTensor) shouldn't hold any
        # memory for the class in question, but it should still
        # advertise the same device as before
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            elem.size(),
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
            # TODO: clone storage aliasing
            dtype=elem.dtype,
            layout=elem.layout,
            device=elem.device,
            requires_grad=elem.requires_grad,
        )
        # ...the real tensor is held as an element on the tensor.
        r.elem = elem
        return r

    def __repr__(self, tensor_contents: Optional[None] = None) -> str:
        if self.grad_fn:
            return f"FlopTensor({self.elem}, grad_fn={self.grad_fn})"
        return f"FlopTensor({self.elem})"

    @classmethod
    def __torch_dispatch__(
        cls,
        func: Callable[..., Any],  # pyre-ignore [2] func can be any func
        types: Tuple[Any],  # pyre-ignore [2]
        args=(),  # pyre-ignore [2]
        kwargs=None,  # pyre-ignore [2]
    ) -> PyTree:
        def unwrap(e: torch.Tensor) -> torch.Tensor:
            return e.elem if isinstance(e, FlopTensor) else e

        # no_dispatch is only needed if you use enable_python_mode.
        # It prevents infinite recursion.
        rs = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
        outs = _normalize_tuple(rs)

        if func in flop_mapping:
            flop_counts = FlopTensor.flop_counts
            flop_count = flop_mapping[func](args, outs)
            for par in FlopTensor.parents:
                # pyre-ignore [58]
                FlopTensor.flop_counts[par][func.__name__] += flop_count
        else:
            logging.debug(f"{func} is not yet supported in FLOPs calculation.")

        def wrap(e: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return FlopTensor(e) if isinstance(e, torch.Tensor) else e

        rs = tree_map(wrap, rs)
        return rs


# pyre-ignore [3]
def _create_backwards_push(name: str) -> Callable[..., Any]:
    class PushState(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            args = tree_map(
                lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args
            )
            if len(args) == 1:
                return args[0]
            return args

        @staticmethod
        def backward(ctx, *grad_outs):
            parents = FlopTensor.parents
            parents.append(name)
            return grad_outs

    # Pyre does not support analyzing classes nested in functions.
    # But this class can't be lifted out of the function as it is a static class
    # using a function parameter.
    # pyre-ignore [16]
    return PushState.apply


# pyre-ignore [3]
def _create_backwards_pop(name: str) -> Callable[..., Any]:
    class PopState(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            args = tree_map(
                lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args
            )
            if len(args) == 1:
                return args[0]
            return args

        @staticmethod
        def backward(ctx, *grad_outs):
            parents = FlopTensor.parents
            assert parents[-1] == name
            parents.pop()
            return grad_outs

    # Pyre does not support analyzing classes nested in functions.
    # But this class can't be lifted out of the function as it is a static class
    # using a function parameter.
    # pyre-ignore [16]
    return PopState.apply


# pyre-ignore [3] Return a callable function
def _enter_module(name: str) -> Callable[..., Any]:
    # pyre-ignore [2, 3]
    def f(module: torch.nn.Module, inputs: Tuple[Any]):
        parents = FlopTensor.parents
        parents.append(name)
        inputs = _normalize_tuple(inputs)
        out = _create_backwards_pop(name)(*inputs)
        return out

    return f


# pyre-ignore [3] Return a callable function
def _exit_module(name: str) -> Callable[..., Any]:
    # pyre-ignore [2, 3]
    def f(module: torch.nn.Module, inputs: Tuple[Any], outputs: Tuple[Any]):
        parents = FlopTensor.parents
        assert parents[-1] == name
        parents.pop()
        outputs = _normalize_tuple(outputs)
        return _create_backwards_push(name)(*outputs)

    return f


def instrument_module(
    mod: torch.nn.Module,
    all_hooks: List[torch.utils.hooks.RemovableHandle],
    par_name: str,
) -> None:
    for name, module in dict(mod.named_children()).items():
        formatted_name = name
        if par_name != "":
            formatted_name = f"{par_name}.{name}"
        all_hooks.append(
            module.register_forward_pre_hook(_enter_module(formatted_name))
        )
        all_hooks.append(module.register_forward_hook(_exit_module(formatted_name)))
        instrument_module(module, all_hooks, formatted_name)


def start_counting() -> None:
    FlopTensor.parents = [""]
    FlopTensor.flop_counts.clear()
