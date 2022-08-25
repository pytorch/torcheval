# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torcheval.metrics.window import WindowedBinaryNormalizedEntropy
from torcheval.utils.test_utils.metric_class_tester import MetricClassTester


class TestWindowedBinaryNormalizedEntropy(MetricClassTester):
    def test_ne_with_valid_input(self) -> None:
        input = torch.tensor([[0.2, 0.3], [0.5, 0.1], [0.3, 0.5], [0.2, 0.4]])
        input_logit = torch.logit(input)
        target = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        weight = torch.tensor([[5.0, 1.0], [2.0, 3.0], [4.0, 7.0], [1.0, 1.0]])

        # disable lifetime
        self.run_class_implementation_tests(
            metric=WindowedBinaryNormalizedEntropy(
                window_size=1, enable_lifetime=False
            ),
            state_names={
                "windowed_total_entropy",
                "windowed_num_examples",
                "windowed_num_positive",
            },
            update_kwargs={"input": input, "target": target},
            compute_result=torch.tensor([153533328963756.2], dtype=torch.float64),
            merge_and_compute_result=torch.tensor(
                [1.477871525897720], dtype=torch.float64
            ),
            num_total_updates=4,
            num_processes=2,
        )

        # without weight and input are probability value
        self.run_class_implementation_tests(
            metric=WindowedBinaryNormalizedEntropy(window_size=1),
            state_names={
                "total_entropy",
                "num_examples",
                "num_positive",
                "windowed_total_entropy",
                "windowed_num_examples",
                "windowed_num_positive",
            },
            update_kwargs={"input": input, "target": target},
            compute_result=(
                torch.tensor([1.046165732800875], dtype=torch.float64),
                torch.tensor([153533328963756.2], dtype=torch.float64),
            ),
            merge_and_compute_result=(
                torch.tensor([1.046165732800875], dtype=torch.float64),
                torch.tensor([1.477871525897720], dtype=torch.float64),
            ),
            num_total_updates=4,
            num_processes=2,
        )

        # with weight and input are probability value
        self.run_class_implementation_tests(
            metric=WindowedBinaryNormalizedEntropy(window_size=1),
            state_names={
                "total_entropy",
                "num_examples",
                "num_positive",
                "windowed_total_entropy",
                "windowed_num_examples",
                "windowed_num_positive",
            },
            update_kwargs={"input": input, "target": target, "weight": weight},
            compute_result=(
                torch.tensor([1.0060274419349144], dtype=torch.float64),
                torch.tensor([153533328963756.2], dtype=torch.float64),
            ),
            merge_and_compute_result=(
                torch.tensor([1.0060274419349144], dtype=torch.float64),
                torch.tensor([0.884474688397802], dtype=torch.float64),
            ),
            num_total_updates=4,
            num_processes=2,
        )

        # without weight and input are logit value
        self.run_class_implementation_tests(
            metric=WindowedBinaryNormalizedEntropy(window_size=1, from_logits=True),
            state_names={
                "total_entropy",
                "num_examples",
                "num_positive",
                "windowed_total_entropy",
                "windowed_num_examples",
                "windowed_num_positive",
            },
            update_kwargs={"input": input_logit, "target": target},
            compute_result=(
                torch.tensor([1.046165732800875], dtype=torch.float64),
                torch.tensor([153533328963756.2], dtype=torch.float64),
            ),
            merge_and_compute_result=(
                torch.tensor([1.046165732800875], dtype=torch.float64),
                torch.tensor([1.477871525897720], dtype=torch.float64),
            ),
            num_total_updates=4,
            num_processes=2,
        )

        # with weight and input are logit value
        self.run_class_implementation_tests(
            metric=WindowedBinaryNormalizedEntropy(window_size=1, from_logits=True),
            state_names={
                "total_entropy",
                "num_examples",
                "num_positive",
                "windowed_total_entropy",
                "windowed_num_examples",
                "windowed_num_positive",
            },
            update_kwargs={"input": input_logit, "target": target, "weight": weight},
            compute_result=(
                torch.tensor([1.0060274419349144], dtype=torch.float64),
                torch.tensor([153533328963756.2], dtype=torch.float64),
            ),
            merge_and_compute_result=(
                torch.tensor([1.0060274419349144], dtype=torch.float64),
                torch.tensor([0.884474688397802], dtype=torch.float64),
            ),
            num_total_updates=4,
            num_processes=2,
        )

        # multi-task
        input_multi_tasks = torch.tensor(
            [
                [[0.2, 0.3], [0.5, 0.1]],
                [[0.3, 0.5], [0.2, 0.4]],
                [[0.6, 0.8], [0.1, 0.5]],
                [[0.7, 0.2], [0.4, 0.7]],
            ]
        )
        input_logit_multi_tasks = torch.logit(input_multi_tasks)
        target_multi_tasks = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[1.0, 0.0], [1.0, 1.0]],
                [[0.0, 0.0], [1.0, 1.0]],
                [[1.0, 1.0], [1.0, 0.0]],
            ]
        )
        weight_multi_tasks = torch.tensor(
            [
                [[5.0, 1.0], [2.0, 3.0]],
                [[4.0, 7.0], [1.0, 1.0]],
                [[6.0, 2.0], [2.0, 8.0]],
                [[6.0, 2.0], [2.0, 1.0]],
            ]
        )

        # without weight and input are probability value
        self.run_class_implementation_tests(
            metric=WindowedBinaryNormalizedEntropy(window_size=1, num_tasks=2),
            state_names={
                "total_entropy",
                "num_examples",
                "num_positive",
                "windowed_total_entropy",
                "windowed_num_examples",
                "windowed_num_positive",
            },
            update_kwargs={"input": input_multi_tasks, "target": target_multi_tasks},
            compute_result=(
                torch.tensor(
                    [1.316725983018726, 1.876157042932679], dtype=torch.float64
                ),
                torch.tensor(
                    [119515548521352.6, 1.529446873602483], dtype=torch.float64
                ),
            ),
            merge_and_compute_result=(
                torch.tensor(
                    [1.316725983018726, 1.876157042932679], dtype=torch.float64
                ),
                torch.tensor(
                    [1.717495735415587, 2.065490803621267], dtype=torch.float64
                ),
            ),
            num_total_updates=4,
            num_processes=2,
        )

        # # without weight and input are logit value
        self.run_class_implementation_tests(
            metric=WindowedBinaryNormalizedEntropy(
                window_size=1, num_tasks=2, from_logits=True
            ),
            state_names={
                "total_entropy",
                "num_examples",
                "num_positive",
                "windowed_total_entropy",
                "windowed_num_examples",
                "windowed_num_positive",
            },
            update_kwargs={
                "input": input_logit_multi_tasks,
                "target": target_multi_tasks,
            },
            compute_result=(
                torch.tensor(
                    [1.316725983018726, 1.876157042932679], dtype=torch.float64
                ),
                torch.tensor(
                    [119515548521352.6, 1.529446873602483], dtype=torch.float64
                ),
            ),
            merge_and_compute_result=(
                torch.tensor(
                    [1.316725983018726, 1.876157042932679], dtype=torch.float64
                ),
                torch.tensor(
                    [1.717495735415587, 2.065490803621267], dtype=torch.float64
                ),
            ),
            num_total_updates=4,
            num_processes=2,
        )

        # with weight and input are probability value
        self.run_class_implementation_tests(
            metric=WindowedBinaryNormalizedEntropy(window_size=1, num_tasks=2),
            state_names={
                "total_entropy",
                "num_examples",
                "num_positive",
                "windowed_total_entropy",
                "windowed_num_examples",
                "windowed_num_positive",
            },
            update_kwargs={
                "input": input_multi_tasks,
                "target": target_multi_tasks,
                "weight": weight_multi_tasks,
            },
            compute_result=(
                torch.tensor(
                    [1.165733397458778, 1.740100161150950], dtype=torch.float64
                ),
                torch.tensor(
                    [81439241662489.02, 1.590199918975681], dtype=torch.float64
                ),
            ),
            merge_and_compute_result=(
                torch.tensor(
                    [1.165733397458778, 1.740100161150950], dtype=torch.float64
                ),
                torch.tensor(
                    [1.201754559517199, 2.223123940833007], dtype=torch.float64
                ),
            ),
            num_total_updates=4,
            num_processes=2,
        )

        # with weight and input are logit value
        self.run_class_implementation_tests(
            metric=WindowedBinaryNormalizedEntropy(
                window_size=1, num_tasks=2, from_logits=True
            ),
            state_names={
                "total_entropy",
                "num_examples",
                "num_positive",
                "windowed_total_entropy",
                "windowed_num_examples",
                "windowed_num_positive",
            },
            update_kwargs={
                "input": input_logit_multi_tasks,
                "target": target_multi_tasks,
                "weight": weight_multi_tasks,
            },
            compute_result=(
                torch.tensor(
                    [1.165733397458778, 1.740100161150950], dtype=torch.float64
                ),
                torch.tensor(
                    [81439241662489.02, 1.590199918975681], dtype=torch.float64
                ),
            ),
            merge_and_compute_result=(
                torch.tensor(
                    [1.165733397458778, 1.740100161150950], dtype=torch.float64
                ),
                torch.tensor(
                    [1.201754559517199, 2.223123940833007], dtype=torch.float64
                ),
            ),
            num_total_updates=4,
            num_processes=2,
        )

    def test_ne_with_invalid_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "`num_tasks` value should be greater"):
            metric = WindowedBinaryNormalizedEntropy(num_tasks=-1)

        metric = WindowedBinaryNormalizedEntropy()
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
