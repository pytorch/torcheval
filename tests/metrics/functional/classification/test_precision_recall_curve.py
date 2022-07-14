# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Optional, Tuple

import torch
from sklearn.metrics import precision_recall_curve
from torch.nn import functional as F
from torcheval.metrics.functional import (
    precision_recall_curve as my_precision_recall_curve,
)
from torcheval.test_utils.metric_class_tester import BATCH_SIZE


class TestPrecisionRecallCurve(unittest.TestCase):
    def _test_helper(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        my_compute_result: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        compute_result = precision_recall_curve(target, input)
        compute_result = [
            torch.tensor(x.copy(), dtype=torch.float32) for x in compute_result
        ]
        for my_tensor, tensor in zip(my_compute_result, compute_result):
            torch.testing.assert_close(
                my_tensor,
                tensor,
                equal_nan=True,
                atol=1e-8,
                rtol=1e-5,
            )

    def _test_precision_recall_curve_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_classes: Optional[int] = None,
    ) -> None:
        my_compute_result = my_precision_recall_curve(
            input, target, num_classes=num_classes
        )
        if num_classes is not None:
            target = F.one_hot(target, num_classes=num_classes)
            for idx in range(num_classes):
                my_compute_result_idx = [
                    my_compute_result[0][idx],
                    my_compute_result[1][idx],
                    my_compute_result[2][idx],
                ]
                self._test_helper(
                    input[:, idx].reshape(1, -1).squeeze(),
                    target[:, idx].reshape(1, -1).squeeze(),
                    tuple(my_compute_result_idx),
                )
        else:
            self._test_helper(input, target, my_compute_result)

    def test_precision_recall_curve_base(self) -> None:
        input = torch.rand(BATCH_SIZE)
        target = torch.randint(high=2, size=(BATCH_SIZE,))
        self._test_precision_recall_curve_with_input(input, target)

        input = torch.rand(BATCH_SIZE)
        self._test_precision_recall_curve_with_input(input, target)

        num_classes = 3
        input = torch.tensor(
            [
                [0.1, 0.2, 0.1],
                [0.4, 0.2, 0.1],
                [0.6, 0.1, 0.2],
                [0.4, 0.2, 0.3],
                [0.6, 0.2, 0.4],
            ]
        )
        target = torch.randint(high=num_classes, size=(5,))
        self._test_precision_recall_curve_with_input(
            input, target, num_classes=num_classes
        )

        num_classes = 3
        input = torch.rand(BATCH_SIZE, num_classes)
        target = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        self._test_precision_recall_curve_with_input(
            input, target, num_classes=num_classes
        )

        num_classes = 5
        input = torch.rand(BATCH_SIZE, num_classes)
        target = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        self._test_precision_recall_curve_with_input(
            input, target, num_classes=num_classes
        )

    def test_precision_recall_curve_label_not_exist(self) -> None:
        num_classes = 4
        input = torch.rand(BATCH_SIZE, num_classes)
        target = torch.randint(high=num_classes - 1, size=(BATCH_SIZE,))
        self._test_precision_recall_curve_with_input(
            input, target, num_classes=num_classes
        )

        num_classes = 8
        input = torch.rand(BATCH_SIZE, num_classes)
        target = torch.randint(high=num_classes - 2, size=(BATCH_SIZE,))
        self._test_precision_recall_curve_with_input(
            input, target, num_classes=num_classes
        )

    def test_precision_recall_curve_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same first dimension, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            my_precision_recall_curve(torch.rand(4, 2), torch.rand(3), num_classes=2)

        with self.assertRaisesRegex(
            ValueError,
            "target should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            my_precision_recall_curve(torch.rand(3, 2), torch.rand(3, 2), num_classes=2)

        with self.assertRaisesRegex(
            ValueError,
            r"input should have shape of \(num_sample,\) or \(num_sample, num_classes\), "
            r"got torch.Size\(\[3, 4\]\) and num_classes=2.",
        ):
            my_precision_recall_curve(torch.rand(3, 4), torch.rand(3), num_classes=2)
