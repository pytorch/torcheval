# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import torch
from sklearn.metrics import accuracy_score
from torcheval.metrics import MultiLabelAccuracy
from torcheval.utils.test_utils.metric_class_tester import (
    BATCH_SIZE,
    MetricClassTester,
    NUM_TOTAL_UPDATES,
)


class TestMultiLabelAccuracy(MetricClassTester):
    def _test_exact_match_with_input(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> None:
        input_np = input.squeeze().numpy().round()
        target_np = target.squeeze().numpy()

        sklearn_result = torch.tensor(
            np.stack([accuracy_score(t, i) for t, i in zip(input_np, target_np)])
        )
        self.run_class_implementation_tests(
            metric=MultiLabelAccuracy(),
            state_names={"num_correct", "num_total"},
            update_kwargs={"input": input, "target": target},
            compute_result=sklearn_result.to(torch.float32).mean(),
        )

    def _test_hamming_with_input(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> None:
        input_np = input.numpy().round()
        target_np = target.numpy()
        sklearn_result = torch.tensor(
            np.stack(
                [
                    accuracy_score(t.flatten(), i.flatten())
                    for t, i in zip(input_np, target_np)
                ]
            )
        )

        self.run_class_implementation_tests(
            metric=MultiLabelAccuracy(criteria="hamming"),
            state_names={"num_correct", "num_total"},
            update_kwargs={"input": input, "target": target},
            compute_result=sklearn_result.to(torch.float32).mean(),
        )

    def test_multi_label_accuracy_exact_match(self) -> None:
        num_classes = 2
        input = torch.randint(0, 2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes))
        target = torch.randint(0, 2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes))

        self._test_exact_match_with_input(input, target)

    def test_multi_label_accuracy_exact_match_with_rounding(self) -> None:
        num_classes = 2
        input = torch.rand(size=(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes))
        target = torch.randint(0, 2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes))

        self._test_exact_match_with_input(input, target)

    def test_multi_label_accuracy_hamming(self) -> None:
        num_classes = 2
        input = torch.randint(0, 2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes))
        target = torch.randint(0, 2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes))

        self._test_hamming_with_input(input, target)

    def test_multi_label_accuracy_hamming_with_rounding(self) -> None:
        num_classes = 2
        input = torch.rand(size=(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes))
        target = torch.randint(0, 2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes))

        self._test_hamming_with_input(input, target)

    def test_accuracy_class_invalid_intialization(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "`criteria` was not in the allowed value of .*, got weighted."
        ):
            MultiLabelAccuracy(criteria="weighted")

    def test_accuracy_class_invalid_input(self) -> None:
        metric = MultiLabelAccuracy()
        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same dimensions, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            metric.update(torch.rand(4, 2), torch.rand(3))
