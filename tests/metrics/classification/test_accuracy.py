# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import torch
from sklearn.metrics import accuracy_score
from torcheval.metrics import Accuracy
from torcheval.test_utils.metric_class_tester import (
    BATCH_SIZE,
    MetricClassTester,
    NUM_TOTAL_UPDATES,
)


class TestAccuracy(MetricClassTester):
    def _test_accuracy_class_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
    ) -> None:
        input_tensors = [t.squeeze() for t in input]
        input_tensors = [
            torch.argmax(t, dim=1) if t.ndim == 2 else t for t in input_tensors
        ]
        target_tensors = [t.squeeze() for t in target]

        target_np = torch.stack(target_tensors).flatten().numpy()
        input_np = torch.stack(input_tensors).flatten().numpy()
        compute_result = torch.tensor(accuracy_score(target_np, input_np)).to(
            torch.float32
        )
        self.run_class_implementation_tests(
            metric=Accuracy(),
            state_names={"num_correct", "num_total"},
            update_kwargs={"input": input, "target": target},
            compute_result=compute_result,
        )

    def test_accuracy_class_base(self) -> None:
        num_classes = 4
        input = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        self._test_accuracy_class_with_input(input, target, num_classes)

        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes)
        self._test_accuracy_class_with_input(input, target, num_classes)

    def test_accuracy_class_average(self) -> None:
        num_classes = 4
        # high=num_classes-1 gives us NaN value for the last class
        input = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        target = torch.randint(
            high=num_classes - 1, size=(NUM_TOTAL_UPDATES, BATCH_SIZE)
        )

        input_flattened = input.flatten()
        target_flattened = target.flatten()
        accuracy_per_class = np.empty(num_classes)
        for i in range(num_classes):
            accuracy_per_class[i] = accuracy_score(
                target_flattened[target_flattened == i].numpy(),
                input_flattened[target_flattened == i].numpy(),
            )

        self.run_class_implementation_tests(
            metric=Accuracy(num_classes=num_classes, average="macro"),
            state_names={"num_correct", "num_total"},
            update_kwargs={"input": input, "target": target},
            compute_result=torch.tensor(
                np.mean(accuracy_per_class[~np.isnan(accuracy_per_class)])
            ).to(torch.float32),
        )

        self.run_class_implementation_tests(
            metric=Accuracy(num_classes=num_classes, average=None),
            state_names={"num_correct", "num_total"},
            update_kwargs={"input": input, "target": target},
            compute_result=torch.tensor(accuracy_per_class).view(-1).to(torch.float32),
        )

    def test_accuracy_class_invalid_intialization(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "`average` was not in the allowed value of .*, got weighted."
        ):
            Accuracy(num_classes=4, average="weighted")

        with self.assertRaisesRegex(
            ValueError,
            "num_classes should be a positive number when average=None. Got num_classes=None",
        ):
            Accuracy(average=None)

    def test_accuracy_class_invalid_input(self) -> None:
        metric = Accuracy()
        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same first dimension, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            metric.update(torch.rand(4, 2), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError, "target should be a one-dimensional tensor, got shape ."
        ):
            metric.update(torch.rand(BATCH_SIZE, 1), torch.rand(BATCH_SIZE, 1))

        with self.assertRaisesRegex(ValueError, "input should have shape"):
            metric.update(torch.rand(BATCH_SIZE, 2, 1), torch.rand(BATCH_SIZE))
