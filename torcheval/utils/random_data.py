# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch


def get_rand_data_binary(
    num_updates: int, num_tasks: int, batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a random binary dataset.

    Args:
        num_updates: the number of calls to update on each rank.
        num_tasks: the number of tasks for the metric.
        batch_size: batch size of the dataset.

    Returns:
        torch.Tensor: random feature data
        torch.Tensor: random targets
    """
    input = torch.rand(size=[num_updates, num_tasks, batch_size])
    targets = torch.randint(low=0, high=2, size=[num_updates, num_tasks, batch_size])
    return input, targets


def get_rand_data_multiclass(
    num_updates: int, num_classes: int, batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a random multiclass dataset.

    Args:
        num_updates: the number of calls to update on each rank.
        num_classes: the number of classes for the dataset.
        batch_size: batch size of the dataset.

    Returns:
        torch.Tensor: random feature data
        torch.Tensor: random targets
    """
    input = torch.rand(size=[num_updates, batch_size, num_classes])
    targets = torch.randint(low=0, high=num_classes, size=[num_updates, batch_size])
    return input, targets
