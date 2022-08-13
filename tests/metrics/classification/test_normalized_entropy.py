# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torcheval.metrics.classification import BinaryNormalizedEntropy
from torcheval.utils.test_utils.metric_class_tester import MetricClassTester


class TestBinaryNormalizedEntropy(MetricClassTester):
    def test_ne_with_valid_input(self) -> None:
        input = torch.tensor([[0.2, 0.3], [0.5, 0.1], [0.3, 0.5], [0.2, 0.4]])
        input_logit = torch.logit(input)
        target = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        weight = torch.tensor([[5.0, 1.0], [2.0, 3.0], [4.0, 7.0], [1.0, 1.0]])

        # without weight and input are probability value
        self.run_class_implementation_tests(
            metric=BinaryNormalizedEntropy(),
            state_names={"total_entropy", "num_examples", "num_positive"},
            update_kwargs={"input": input, "target": target},
            compute_result=torch.tensor(1.046165732800875, dtype=torch.float64),
            num_total_updates=4,
            num_processes=2,
        )

        # with weight and input are probability value
        self.run_class_implementation_tests(
            metric=BinaryNormalizedEntropy(),
            state_names={"total_entropy", "num_examples", "num_positive"},
            update_kwargs={"input": input, "target": target, "weight": weight},
            compute_result=torch.tensor(1.0060274419349144, dtype=torch.float64),
            num_total_updates=4,
            num_processes=2,
        )

        # without weight and input are logit value
        self.run_class_implementation_tests(
            metric=BinaryNormalizedEntropy(from_logits=True),
            state_names={"total_entropy", "num_examples", "num_positive"},
            update_kwargs={"input": input_logit, "target": target},
            compute_result=torch.tensor(1.046165732800875, dtype=torch.float64),
            num_total_updates=4,
            num_processes=2,
        )

        # with weight and input are logit value
        self.run_class_implementation_tests(
            metric=BinaryNormalizedEntropy(from_logits=True),
            state_names={"total_entropy", "num_examples", "num_positive"},
            update_kwargs={"input": input_logit, "target": target, "weight": weight},
            compute_result=torch.tensor(1.0060274419349144, dtype=torch.float64),
            num_total_updates=4,
            num_processes=2,
        )

    def test_ne_with_invalid_input(self) -> None:
        metric = BinaryNormalizedEntropy()
        with self.assertRaisesRegex(ValueError, "is different from `target` shape"):
            metric.update(torch.rand((5,)), torch.randint(0, 2, (3,)))

        with self.assertRaisesRegex(ValueError, "is different from `input` shape"):
            metric.update(
                torch.rand((5,)),
                torch.randint(0, 2, (5,)),
                weight=torch.randint(0, 20, (3,)),
            )
        with self.assertRaisesRegex(
            ValueError,
            "`input` should be probability",
        ):
            metric.update(
                torch.rand((5,)) * 10.0,
                torch.randint(0, 2, (5,)),
            )
