# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch

from sklearn.metrics import precision_recall_curve
from torch.nn import functional as F

from torcheval.metrics import PrecisionRecallCurve
from torcheval.test_utils.metric_class_tester import (
    BATCH_SIZE,
    MetricClassTester,
    NUM_TOTAL_UPDATES,
)


class TestPrecisionRecallCurve(MetricClassTester):
    def _test_helper(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> List[torch.Tensor]:
        compute_result = precision_recall_curve(target, input)
        compute_result = [
            torch.tensor(x.copy(), dtype=torch.float32) for x in compute_result
        ]
        return compute_result

    def _test_precision_recall_curve_class_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_classes: Optional[int] = None,
    ) -> None:
        input_tensors = input.reshape(-1, 1)
        target_tensors = target.reshape(1, -1).squeeze()
        if num_classes is not None:
            input_tensors = input.reshape(-1, num_classes)
            target_tensors = target.reshape(1, -1).squeeze()
            precision, recall, thresholds = [], [], []
            target_tensors = F.one_hot(target_tensors, num_classes=num_classes)
            for idx in range(num_classes):
                p, r, t = self._test_helper(
                    input_tensors[:, idx],
                    target_tensors[:, idx],
                )
                precision.append(p)
                recall.append(r)
                thresholds.append(t)
        else:
            precision, recall, thresholds = self._test_helper(
                input_tensors, target_tensors
            )

        compute_result = (precision, recall, thresholds)
        self.run_class_implementation_tests(
            metric=PrecisionRecallCurve(num_classes=num_classes),
            state_names={"inputs", "targets"},
            update_kwargs={
                "input": input,
                "target": target,
            },
            compute_result=compute_result,
        )

    def test_precision_recall_curve_class_base(self) -> None:
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        self._test_precision_recall_curve_class_with_input(input, target)

        num_classes = 4
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes)
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        self._test_precision_recall_curve_class_with_input(
            input, target, num_classes=num_classes
        )

    def test_precision_recall_curve_class_label_not_exist(self) -> None:

        num_classes = 4
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes)
        target = torch.randint(
            high=num_classes - 1, size=(NUM_TOTAL_UPDATES, BATCH_SIZE)
        )
        self._test_precision_recall_curve_class_with_input(
            input, target, num_classes=num_classes
        )

        num_classes = 8
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes)
        target = torch.randint(
            high=num_classes - 2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE)
        )
        self._test_precision_recall_curve_class_with_input(
            input, target, num_classes=num_classes
        )

    def test_precision_recall_curve_class_update_input_shape_different(self) -> None:
        num_classes = 2
        update_input = [
            torch.rand(5),
            torch.rand(8),
            torch.rand(2),
            torch.rand(5),
        ]

        update_target = [
            torch.randint(high=num_classes, size=(5,)),
            torch.randint(high=num_classes, size=(8,)),
            torch.randint(high=num_classes, size=(2,)),
            torch.randint(high=num_classes, size=(5,)),
        ]
        compute_result = precision_recall_curve(
            torch.cat(update_target, dim=0),
            torch.cat(update_input, dim=0),
        )

        compute_result = [
            torch.tensor(c.copy(), dtype=torch.float32) for c in compute_result
        ]

        self.run_class_implementation_tests(
            metric=PrecisionRecallCurve(),
            state_names={"inputs", "targets"},
            update_kwargs={
                "input": update_input,
                "target": update_target,
            },
            compute_result=tuple(compute_result),
            num_total_updates=4,
            num_processes=2,
        )

    def test_precision_recall_curve_class_invalid_input(self) -> None:
        metric = PrecisionRecallCurve(num_classes=4)
        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same first dimension, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            metric.update(torch.rand(4, 2), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "target should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            metric.update(torch.rand(3, 2), torch.rand(3, 2))

        with self.assertRaisesRegex(
            ValueError,
            r"input should have shape of \(num_sample,\) or \(num_sample, num_classes\), "
            r"got torch.Size\(\[3, 4\]\) and num_classes=2.",
        ):
            metric = PrecisionRecallCurve(num_classes=2)
            metric.update(torch.rand(3, 4), torch.rand(3))
