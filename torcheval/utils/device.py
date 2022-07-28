# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import defaultdict
from dataclasses import fields, is_dataclass
from typing import Any, Mapping, TypeVar

import torch
from typing_extensions import Protocol, runtime_checkable


def get_device_from_env() -> torch.device:
    """Function that gets the torch.device based on the current environment.

    This currently supports only CPU and GPU devices. If CUDA is available, this function also sets the CUDA device.

    Within a distributed context, this function relies on the ``LOCAL_RANK` environment variable
    to be made available by the program launcher for setting the appropriate device index.

    Raises:
        RuntimeError
            If ``LOCAL_RANK`` is outside the range of available GPU devices.
    """
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                "The local rank is larger than the number of available GPUs."
            )
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


T = TypeVar("T")
TSelf = TypeVar("TSelf")


@runtime_checkable
class _CopyableData(Protocol):
    def to(self: TSelf, device: torch.device, *args: Any, **kwargs: Any) -> TSelf:
        """Copy data to the specified device"""
        ...


def _is_named_tuple(x: T) -> bool:
    return isinstance(x, tuple) and hasattr(x, "_asdict") and hasattr(x, "_fields")


def copy_data_to_device(data: T, device: torch.device, *args: Any, **kwargs: Any) -> T:
    """Function that recursively copies data to a torch.device.

    Args:
        data: The data to copy to device
        device: The device to which the data should be copied
        args: positional arguments that will be passed to the `to` call
        kwargs: keyword arguments that will be passed to the `to` call

    Returns:
        The data on the correct device
    """

    # Redundant isinstance(data, tuple) check is required here to make pyre happy
    if _is_named_tuple(data) and isinstance(data, tuple):
        return type(data)(
            **copy_data_to_device(data._asdict(), device, *args, **kwargs)
        )
    elif isinstance(data, (list, tuple)):
        return type(data)(copy_data_to_device(e, device, *args, **kwargs) for e in data)
    elif isinstance(data, defaultdict):
        return type(data)(
            data.default_factory,
            {
                k: copy_data_to_device(v, device, *args, **kwargs)
                for k, v in data.items()
            },
        )
    elif isinstance(data, Mapping):
        return type(data)(
            {
                k: copy_data_to_device(v, device, *args, **kwargs)
                for k, v in data.items()
            }
        )
    elif is_dataclass(data) and not isinstance(data, type):
        new_data_class = type(data)(
            **{
                field.name: copy_data_to_device(
                    getattr(data, field.name), device, *args, **kwargs
                )
                for field in fields(data)
                if field.init
            }
        )
        for field in fields(data):
            if not field.init:
                setattr(
                    new_data_class,
                    field.name,
                    copy_data_to_device(
                        getattr(data, field.name), device, *args, **kwargs
                    ),
                )
        return new_data_class
    elif isinstance(data, _CopyableData):
        return data.to(device, *args, **kwargs)
    return data
