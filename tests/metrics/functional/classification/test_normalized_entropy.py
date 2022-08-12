# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torcheval.metrics.functional import binary_normalized_entropy


class TestBinaryNormalizedEntropy(unittest.TestCase):
    def test_ne_with_valid_input(self) -> None:
        input = torch.tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2, 0.4])
        input_logit = torch.logit(input)
        target = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
        weight = torch.tensor([5.0, 1.0, 2.0, 3.0, 4.0, 7.0, 1.0, 1.0])

        # without weight and input are probability value
        torch.testing.assert_close(
            binary_normalized_entropy(input, target, weight=None, from_logits=False),
            torch.tensor(1.046165732800875, dtype=torch.float64),
        )

        # with weight and input are probability value
        torch.testing.assert_close(
            binary_normalized_entropy(input, target, weight, from_logits=False),
            torch.tensor(1.0060274419349144, dtype=torch.float64),
        )

        # without weight and input are logit value
        torch.testing.assert_close(
            binary_normalized_entropy(
                input_logit, target, weight=None, from_logits=True
            ),
            torch.tensor(1.046165732800875, dtype=torch.float64),
        )

        # with weight and input are logit value
        torch.testing.assert_close(
            binary_normalized_entropy(input_logit, target, weight, from_logits=True),
            torch.tensor(1.0060274419349144, dtype=torch.float64),
        )

    def test_ne_with_invalid_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "is different from `target` shape"):
            binary_normalized_entropy(torch.rand((5,)), torch.randint(0, 2, (3,)))

        with self.assertRaisesRegex(ValueError, "is different from `input` shape"):
            binary_normalized_entropy(
                torch.rand((5,)), torch.randint(0, 2, (5,)), torch.randint(0, 20, (3,))
            )
        with self.assertRaisesRegex(
            ValueError,
            "`input` should be probability",
        ):
            binary_normalized_entropy(
                torch.rand((5,)) * 10.0,
                torch.randint(0, 2, (5,)),
                weight=None,
                from_logits=False,
            )
