# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np

import torch
from sklearn.metrics import accuracy_score
from torcheval.metrics.functional import accuracy
from torcheval.test_utils.metric_class_tester import BATCH_SIZE


class TestAccuracy(unittest.TestCase):
    def _test_accuracy_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
    ) -> None:
        target_np = target.squeeze().flatten().numpy()
        input_squeezed = input.squeeze()
        input_label_ids = (
            torch.argmax(input_squeezed, dim=1)
            if input_squeezed.ndim == 2
            else input_squeezed
        )
        input_np = input_label_ids.flatten().numpy()
        compute_result = torch.tensor(accuracy_score(target_np, input_np)).to(
            torch.float32
        )
        torch.testing.assert_close(
            accuracy(input, target),
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_accuracy_base(self) -> None:
        num_classes = 4
        input = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        target = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        self._test_accuracy_with_input(input, target, num_classes)

        y_score = torch.rand(BATCH_SIZE, num_classes)
        self._test_accuracy_with_input(y_score, target, num_classes)

    def test_accuracy_class_average(self) -> None:
        num_classes = 4
        # high=num_classes-1 gives us NaN value for the last class
        input = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        target = torch.randint(high=num_classes - 1, size=(BATCH_SIZE,))

        input_flattened = input.flatten()
        target_flattened = target.flatten()
        accuracy_per_class = np.empty(num_classes)
        for i in range(num_classes):
            accuracy_per_class[i] = accuracy_score(
                target_flattened[target_flattened == i].numpy(),
                input_flattened[target_flattened == i].numpy(),
            )

        torch.testing.assert_close(
            accuracy(input, target, average="macro", num_classes=num_classes),
            torch.tensor(np.mean(accuracy_per_class[~np.isnan(accuracy_per_class)])).to(
                torch.float32
            ),
            atol=1e-8,
            rtol=1e-5,
        )

        torch.testing.assert_close(
            accuracy(input, target, average=None, num_classes=num_classes),
            torch.tensor(accuracy_per_class).view(-1).to(torch.float32),
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_accuracy_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "`average` was not in the allowed value of .*, got weighted."
        ):
            accuracy(
                torch.randint(high=4, size=(BATCH_SIZE,)),
                torch.randint(high=4, size=(BATCH_SIZE,)),
                num_classes=4,
                average="weighted",
            )

        with self.assertRaisesRegex(
            ValueError,
            "num_classes should be a positive number when average=None. Got num_classes=None",
        ):
            accuracy(
                torch.randint(high=4, size=(BATCH_SIZE,)),
                torch.randint(high=4, size=(BATCH_SIZE,)),
                average=None,
            )

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same first dimension, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            accuracy(torch.rand(4, 2), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError, "target should be a one-dimensional tensor, got shape ."
        ):
            accuracy(torch.rand(BATCH_SIZE, 1), torch.rand(BATCH_SIZE, 1))

        with self.assertRaisesRegex(ValueError, "input should have shape"):
            accuracy(torch.rand(BATCH_SIZE, 2, 1), torch.rand(BATCH_SIZE))
