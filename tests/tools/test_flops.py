#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torchvision.models as models
from torcheval.tools.flops import FlopTensor, instrument_module, start_counting


class ModuleSummaryTest(unittest.TestCase):
    def test_torch_operations(self) -> None:
        """Make sure FLOPs calculation works for a single operations."""
        inp = torch.randn(10, 4, 5)
        bmm_mat = torch.randn(10, 5, 7)
        mm_mat = torch.randn(7, 3)

        inp = FlopTensor(inp)
        start_counting()
        res = inp.bmm(bmm_mat).matmul(mm_mat)

        self.assertEqual(res.shape[0], 10)
        self.assertEqual(res.shape[1], 4)
        self.assertEqual(res.shape[2], 3)

        self.assertEqual(
            FlopTensor.flop_counts[""].get("bmm.default", 0)
            + FlopTensor.flop_counts[""].get("bmm", 0),
            1400,
        )
        self.assertEqual(
            FlopTensor.flop_counts[""].get("mm.default", 0)
            + FlopTensor.flop_counts[""].get("mm", 0),
            840,
        )

        inp = torch.randn(10, 4, 5)
        # pyre-fixme[28]: Unexpected keyword argument `requires_grad`.
        inp = torch.autograd.Variable(inp, requires_grad=True)

        # pyre-fixme[6]: For 1st param expected `Tensor` but got `Variable`.
        inp = FlopTensor(inp)
        start_counting()
        res = inp.bmm(bmm_mat).matmul(mm_mat)
        res.mean().backward()

        self.assertEqual(res.shape[0], 10)
        self.assertEqual(res.shape[1], 4)
        self.assertEqual(res.shape[2], 3)

        self.assertEqual(
            FlopTensor.flop_counts[""].get("bmm.default", 0)
            + FlopTensor.flop_counts[""].get("bmm", 0),
            2800,
        )
        self.assertEqual(
            FlopTensor.flop_counts[""].get("mm.default", 0)
            + FlopTensor.flop_counts[""].get("mm", 0),
            1680,
        )

    def test_torch_linear_layer(self) -> None:
        """Make sure FLOPs calculation works for a module consists of linear layers."""
        lnn = torch.nn.Sequential(
            torch.nn.Sequential(torch.nn.Linear(10, 70), torch.nn.Linear(70, 5)),
            torch.nn.Linear(5, 1),
        )
        inp = torch.randn(1, 10)
        inp = FlopTensor(inp)

        all_hooks = []
        instrument_module(lnn, all_hooks, "")
        self.assertEqual(len(all_hooks), 8)

        start_counting()
        res = lnn(inp)
        self.assertEqual(
            FlopTensor.flop_counts[""].get("addmm.default", 0)
            + FlopTensor.flop_counts[""].get("addmm", 0),
            1055,
        )
        self.assertEqual(
            FlopTensor.flop_counts["0"].get("addmm.default", 0)
            + FlopTensor.flop_counts["0"].get("addmm", 0),
            1050,
        )
        self.assertEqual(
            FlopTensor.flop_counts["0.0"].get("addmm.default", 0)
            + FlopTensor.flop_counts["0.0"].get("addmm", 0),
            700,
        )
        self.assertEqual(
            FlopTensor.flop_counts["0.1"].get("addmm.default", 0)
            + FlopTensor.flop_counts["0.1"].get("addmm", 0),
            350,
        )
        self.assertEqual(
            FlopTensor.flop_counts["1"].get("addmm.default", 0)
            + FlopTensor.flop_counts["1"].get("addmm", 0),
            5,
        )
        start_counting()
        res.backward()
        self.assertEqual(
            FlopTensor.flop_counts[""].get("mm.default", 0)
            + FlopTensor.flop_counts[""].get("mm", 0),
            1410,
        )
        self.assertEqual(
            FlopTensor.flop_counts["0"].get("mm.default", 0)
            + FlopTensor.flop_counts["0"].get("mm", 0),
            1400,
        )
        self.assertEqual(
            FlopTensor.flop_counts["0.0"].get("mm.default", 0)
            + FlopTensor.flop_counts["0.0"].get("mm", 0),
            700,
        )
        self.assertEqual(
            FlopTensor.flop_counts["0.1"].get("mm.default", 0)
            + FlopTensor.flop_counts["0.1"].get("mm", 0),
            700,
        )
        self.assertEqual(
            FlopTensor.flop_counts["1"].get("mm.default", 0)
            + FlopTensor.flop_counts["1"].get("mm", 0),
            10,
        )

    def test_torch_pretrained_module(self) -> None:
        """Make sure FLOPs calculation works for a resnet18."""
        # pyre-fixme[16]: Module `models` has no attribute `resnet18`.
        mod = models.resnet18()
        inp = torch.randn(1, 3, 224, 224)
        all_hooks = []
        instrument_module(mod, all_hooks, "")
        # Hooks should be 2 * number of modules minus 2 (2 for the model itself)
        self.assertEqual(len(all_hooks), 2 * len(list(mod.modules())) - 2)

        inp = FlopTensor(inp)
        start_counting()
        res = mod(inp)

        self.assertEqual(
            FlopTensor.flop_counts[""].get("convolution.default", 0)
            + FlopTensor.flop_counts[""].get("convolution", 0),
            1813561344,
        )
        self.assertEqual(
            FlopTensor.flop_counts[""].get("addmm.default", 0)
            + FlopTensor.flop_counts[""].get("addmm", 0),
            512000,
        )
        self.assertEqual(
            FlopTensor.flop_counts["conv1"].get("convolution.default", 0)
            + FlopTensor.flop_counts["conv1"].get("convolution", 0),
            118013952,
        )
        self.assertEqual(
            FlopTensor.flop_counts["fc"].get("addmm.default", 0)
            + FlopTensor.flop_counts["fc"].get("addmm", 0),
            512000,
        )

        start_counting()
        res.mean().backward()

        self.assertEqual(
            FlopTensor.flop_counts[""].get("convolution_backward.default", 0)
            + FlopTensor.flop_counts[""].get("convolution_backward", 0),
            3509108736,
        )
        self.assertEqual(
            FlopTensor.flop_counts[""].get("mm.default", 0)
            + FlopTensor.flop_counts[""].get("mm", 0),
            1024000,
        )
        self.assertEqual(
            FlopTensor.flop_counts["layer1"].get("convolution_backward.default", 0)
            + FlopTensor.flop_counts["layer1"].get("convolution_backward", 0),
            924844032,
        )
        self.assertEqual(
            FlopTensor.flop_counts["fc"].get("mm.default", 0)
            + FlopTensor.flop_counts["fc"].get("mm", 0),
            1024000,
        )
