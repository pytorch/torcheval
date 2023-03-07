#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torcheval.utils import get_rand_data_binary, get_rand_data_multiclass


class RandomDataTest(unittest.TestCase):
    cuda_avail: bool = torch.cuda.is_available()

    def test_get_rand_data_binary(self) -> None:
        input, targets = get_rand_data_binary(num_updates=2, num_tasks=5, batch_size=10)
        self.assertEqual(input.size(), targets.size())

    def test_get_rand_data_multiclass(self) -> None:
        input, targets = get_rand_data_multiclass(
            num_updates=2, num_classes=5, batch_size=10
        )
        self.assertEqual(input.size(), torch.Size([2, 10, 5]))
        self.assertTrue(torch.all(torch.lt(targets, 5)))

    @unittest.skipUnless(
        condition=cuda_avail, reason="This test needs a GPU host to run."
    )
    def test_get_rand_data_binary_GPU(self) -> None:
        device = torch.device("cuda")
        input, targets = get_rand_data_binary(
            num_updates=2, num_tasks=5, batch_size=10, device=device
        )
        self.assertEqual(input.size(), targets.size())
        self.assertTrue(input.is_cuda)
        self.assertTrue(targets.is_cuda)

    @unittest.skipUnless(
        condition=cuda_avail, reason="This test needs a GPU host to run."
    )
    def test_get_rand_data_multiclass_GPU(self) -> None:
        device = torch.device("cuda")
        input, targets = get_rand_data_multiclass(
            num_updates=2, num_classes=5, batch_size=10, device=device
        )
        self.assertEqual(input.size(), torch.Size([2, 10, 5]))
        self.assertTrue(torch.all(torch.lt(targets, 5)))
        self.assertTrue(input.is_cuda)
        self.assertTrue(targets.is_cuda)
