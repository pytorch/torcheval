# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple, Union

import torch

from torcheval.metrics import (
    BinaryBinnedPrecisionRecallCurve,
    MulticlassBinnedPrecisionRecallCurve,
)
from torcheval.metrics.functional import (
    binary_binned_precision_recall_curve,
    multiclass_binned_precision_recall_curve,
)
from torcheval.utils.test_utils.metric_class_tester import (
    BATCH_SIZE,
    MetricClassTester,
    NUM_TOTAL_UPDATES,
)


class TestBinaryBinnedPrecisionRecallCurve(MetricClassTester):
    def _test_binary_binned_precision_recall_curve_class_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        threshold: Union[int, List[float], torch.Tensor],
    ) -> None:

        compute_result = binary_binned_precision_recall_curve(
            input.reshape(-1), target.reshape(-1), threshold=threshold
        )

        self.run_class_implementation_tests(
            metric=BinaryBinnedPrecisionRecallCurve(threshold=threshold),
            state_names={"num_tp", "num_fp", "num_fn", "threshold"},
            update_kwargs={
                "input": input,
                "target": target,
            },
            compute_result=compute_result,
        )

    def test_binary_binned_precision_recall_curve_class_base(self) -> None:
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        threshold = 200
        self._test_binary_binned_precision_recall_curve_class_with_input(
            input, target, threshold
        )

        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        threshold = torch.tensor(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        self._test_binary_binned_precision_recall_curve_class_with_input(
            input, target, threshold
        )

        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        threshold = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self._test_binary_binned_precision_recall_curve_class_with_input(
            input, target, threshold
        )

    def test_binary_binned_precision_recall_curve_class_update_input_shape_different(
        self,
    ) -> None:
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

        compute_result = binary_binned_precision_recall_curve(
            torch.cat(update_input, dim=0),
            torch.cat(update_target, dim=0),
        )

        self.run_class_implementation_tests(
            metric=BinaryBinnedPrecisionRecallCurve(),
            state_names={"num_tp", "num_fp", "num_fn", "threshold"},
            update_kwargs={
                "input": update_input,
                "target": update_target,
            },
            compute_result=compute_result,
            num_total_updates=4,
            num_processes=2,
        )

    def test_binary_binned_precision_recall_curve_class_invalid_input(self) -> None:
        metric = BinaryBinnedPrecisionRecallCurve(
            threshold=torch.tensor([0.1, 0.5, 0.9])
        )
        with self.assertRaisesRegex(
            ValueError,
            "input should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            metric.update(torch.rand(3, 2), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "target should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            metric.update(torch.rand(3), torch.rand(3, 2))

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same shape, "
            r"got shapes torch.Size\(\[4\]\) and torch.Size\(\[3\]\).",
        ):
            metric.update(torch.rand(4), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError, "The `threshold` should be a sorted array."
        ):
            BinaryBinnedPrecisionRecallCurve(threshold=torch.tensor([0.1, 0.9, 0.5]))

        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            BinaryBinnedPrecisionRecallCurve(threshold=torch.tensor([-0.1, 0.5, 0.9]))

        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            BinaryBinnedPrecisionRecallCurve(
                threshold=torch.tensor([0.1, 0.2, 0.5, 1.7])
            )


class TestMulticlassBinnedPrecisionRecallCurve(MetricClassTester):
    def _test_multiclass_binned_precision_recall_curve_class_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        compute_result: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor],
        num_classes: int,
        threshold: Union[int, List[float], torch.Tensor],
    ) -> None:
        self.run_class_implementation_tests(
            metric=MulticlassBinnedPrecisionRecallCurve(
                num_classes=num_classes, threshold=threshold
            ),
            state_names={"num_tp", "num_fp", "num_fn", "threshold"},
            update_kwargs={
                "input": input,
                "target": target,
            },
            compute_result=compute_result,
        )

    def test_multiclass_binned_precision_recall_curve_class_base(self) -> None:
        num_classes = 3
        input = torch.tensor(
            [
                [[0.1, 0.2, 0.1]],
                [[0.4, 0.2, 0.1]],
                [[0.6, 0.1, 0.2]],
                [[0.4, 0.2, 0.3]],
            ]
        )
        target = torch.tensor([[0], [1], [2], [1]])
        threshold = 3
        compute_result = (
            [
                torch.tensor([0.25, 0.0, 1.0, 1.0]),
                torch.tensor([0.5, 1.0, 1.0, 1.0]),
                torch.tensor([0.25, 1.0, 1.0, 1.0]),
            ],
            [
                torch.tensor([1.0, 0.0, 0.0, 0.0]),
                torch.tensor([1.0, 0.0, 0.0, 0.0]),
                torch.tensor([1.0, 0.0, 0.0, 0.0]),
            ],
            torch.tensor([0.0000, 0.5000, 1.0000]),
        )

        self.run_class_implementation_tests(
            metric=MulticlassBinnedPrecisionRecallCurve(
                num_classes=num_classes, threshold=threshold
            ),
            state_names={"num_tp", "num_fp", "num_fn", "threshold"},
            update_kwargs={
                "input": input,
                "target": target,
            },
            compute_result=compute_result,
            num_total_updates=4,
            num_processes=2,
        )

        num_classes = 3
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes)
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        threshold = 10

        precision, recall, thresholds = multiclass_binned_precision_recall_curve(
            input.reshape(-1, num_classes),
            target.reshape(-1),
            num_classes=num_classes,
            threshold=threshold,
        )

        compute_result = (
            [t.detach().clone() for t in precision],
            [t.detach().clone() for t in recall],
            thresholds,
        )

        self._test_multiclass_binned_precision_recall_curve_class_with_input(
            input,
            target,
            compute_result=compute_result,
            num_classes=num_classes,
            threshold=threshold,
        )

        num_classes = 5
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes)
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        threshold = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        precision, recall, thresholds = multiclass_binned_precision_recall_curve(
            input.reshape(-1, num_classes),
            target.reshape(-1),
            num_classes=num_classes,
            threshold=threshold,
        )

        compute_result = (
            [t.detach().clone() for t in precision],
            [t.detach().clone() for t in recall],
            thresholds,
        )

        self._test_multiclass_binned_precision_recall_curve_class_with_input(
            input,
            target,
            compute_result=compute_result,
            num_classes=num_classes,
            threshold=threshold,
        )

        num_classes = 5
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes)
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        precision, recall, thresholds = multiclass_binned_precision_recall_curve(
            input.reshape(-1, num_classes),
            target.reshape(-1),
            num_classes=num_classes,
            threshold=threshold,
        )

        compute_result = (
            [t.detach().clone() for t in precision],
            [t.detach().clone() for t in recall],
            thresholds,
        )

        self._test_multiclass_binned_precision_recall_curve_class_with_input(
            input,
            target,
            compute_result=compute_result,
            num_classes=num_classes,
            threshold=threshold,
        )

    def test_multiclass_binned_precision_recall_curve_class_update_input_shape_different(
        self,
    ) -> None:
        num_classes = 10
        update_input = [
            torch.rand(5, num_classes),
            torch.rand(8, num_classes),
            torch.rand(2, num_classes),
            torch.rand(5, num_classes),
        ]

        update_target = [
            torch.randint(high=num_classes, size=(5,)),
            torch.randint(high=num_classes, size=(8,)),
            torch.randint(high=num_classes, size=(2,)),
            torch.randint(high=num_classes, size=(5,)),
        ]

        threshold = 10

        precision, recall, thresholds = multiclass_binned_precision_recall_curve(
            torch.cat(update_input, dim=0),
            torch.cat(update_target, dim=0),
            num_classes=num_classes,
            threshold=threshold,
        )

        compute_result = (
            [t.detach().clone() for t in precision],
            [t.detach().clone() for t in recall],
            thresholds,
        )

        self.run_class_implementation_tests(
            metric=MulticlassBinnedPrecisionRecallCurve(
                num_classes=num_classes, threshold=threshold
            ),
            state_names={"num_tp", "num_fp", "num_fn", "threshold"},
            update_kwargs={
                "input": update_input,
                "target": update_target,
            },
            compute_result=compute_result,
            num_total_updates=4,
            num_processes=2,
        )

    def test_multiclass_binned_precision_recall_curve_class_invalid_input(self) -> None:
        metric = MulticlassBinnedPrecisionRecallCurve(num_classes=4)
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
            r"input should have shape of \(num_sample, num_classes\), "
            r"got torch.Size\(\[3, 4\]\) and num_classes=2.",
        ):
            metric = MulticlassBinnedPrecisionRecallCurve(num_classes=2)
            metric.update(torch.rand(3, 4), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError, "The `threshold` should be a sorted array."
        ):
            MulticlassBinnedPrecisionRecallCurve(
                num_classes=5, threshold=torch.tensor([0.1, 0.9, 0.5])
            )

        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            MulticlassBinnedPrecisionRecallCurve(
                num_classes=5, threshold=torch.tensor([-0.1, 0.5, 0.9])
            )

        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            MulticlassBinnedPrecisionRecallCurve(
                num_classes=5, threshold=torch.tensor([0.1, 0.2, 0.5, 1.7])
            )
