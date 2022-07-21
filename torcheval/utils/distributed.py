#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional

import torch
import torch.distributed as dist


class PGWrapper:
    """
    A wrapper around ProcessGroup that allows collectives to be issued in a
    consistent fashion regardless of the following scenarios:

        pg is None, distributed is initialized:     use WORLD as pg
        pg is None, distributed is not initialized: single process app
        pg is not None:                             use pg
    """

    def __init__(self, pg: Optional[dist.ProcessGroup]) -> None:
        if pg is None and dist.is_initialized():
            self.pg: Optional[dist.ProcessGroup] = dist.group.WORLD
        else:
            self.pg: Optional[dist.ProcessGroup] = pg

    def get_rank(self) -> int:
        if self.pg is None:
            return 0
        return dist.get_rank(group=self.pg)

    def get_world_size(self) -> int:
        if self.pg is None:
            return 1
        return dist.get_world_size(group=self.pg)

    def barrier(self) -> None:
        if self.pg is None:
            return
        dist.barrier(group=self.pg)

    # pyre-ignore[2]: obj can have arbitrary types
    def broadcast_object_list(self, obj_list: List[Any], src: int = 0) -> None:
        if self.pg is None:
            return
        dist.broadcast_object_list(obj_list, src=src, group=self.pg)

    # pyre-ignore[2]: obj can have arbitrary types
    def all_gather_object(self, obj_list: List[Any], obj: Any) -> None:
        if self.pg is None:
            obj_list[0] = obj
            return
        dist.all_gather_object(obj_list, obj, group=self.pg)


def get_process_group_backend_from_device(device: torch.device) -> str:
    """Function that gets the defaut process group backend from the device."""
    return "nccl" if device.type == "cuda" else "gloo"
