# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from sklearn.metrics import accuracy_score
from torcheval.metrics.functional import multi_label_accuracy
from torcheval.utils.test_utils.metric_class_tester import BATCH_SIZE


class TestMultiLabelAccuracy(unittest.TestCase):
    def _test_exact_match_with_input(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> None:
        input_np = input.numpy().round()
        target_np = target.squeeze().numpy()
        sklearn_result = torch.tensor(accuracy_score(target_np, input_np)).to(
            torch.float32
        )

        torch.testing.assert_close(
            multi_label_accuracy(input, target),
            sklearn_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def _test_hamming_with_input(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> None:
        input_np = input.numpy().round()
        target_np = target.numpy()
        sklearn_result = torch.tensor(
            accuracy_score(target_np.flatten(), input_np.flatten())
        ).to(torch.float32)

        torch.testing.assert_close(
            multi_label_accuracy(input, target, criteria="hamming"),
            sklearn_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_multi_label_accuracy_exact_match(self) -> None:
        num_classes = 2
        input = torch.randint(0, 2, size=(BATCH_SIZE, num_classes))
        target = torch.randint(0, 2, size=(BATCH_SIZE, num_classes))

        self._test_exact_match_with_input(input, target)

    def test_multi_label_accuracy_exact_match_with_rounding(self) -> None:
        num_classes = 2
        input = torch.rand(size=(BATCH_SIZE, num_classes))
        target = torch.randint(0, 2, size=(BATCH_SIZE, num_classes))

        self._test_exact_match_with_input(input, target)

    def test_multi_label_accuracy_hamming(self) -> None:
        num_classes = 2
        input = torch.randint(0, 2, size=(BATCH_SIZE, num_classes))
        target = torch.randint(0, 2, size=(BATCH_SIZE, num_classes))

        self._test_hamming_with_input(input, target)

    def test_multi_label_accuracy_hamming_with_rounding(self) -> None:
        num_classes = 2
        input = torch.rand(size=(BATCH_SIZE, num_classes))
        target = torch.randint(0, 2, size=(BATCH_SIZE, num_classes))

        self._test_hamming_with_input(input, target)

    def test_multi_label_accuracy_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "`criteria` was not in the allowed value of .*, got weighted."
        ):
            multi_label_accuracy(
                torch.randint(0, 2, size=(BATCH_SIZE, 4)),
                torch.randint(0, 2, size=(BATCH_SIZE, 4)),
                criteria="weighted",
            )

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same dimensions, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            multi_label_accuracy(torch.rand(4, 2), torch.rand(3))
