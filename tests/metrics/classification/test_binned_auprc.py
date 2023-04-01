# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple, Union

import torch
from torcheval.metrics import BinaryBinnedAUPRC
from torcheval.utils.test_utils.metric_class_tester import (
    BATCH_SIZE,
    MetricClassTester,
    NUM_TOTAL_UPDATES,
)


class TestBinaryBinnedAUPRC(MetricClassTester):
    def _test_auprc_class_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_tasks: int,
        threshold: Union[int, List[float], torch.Tensor],
        compute_result: Tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        self.run_class_implementation_tests(
            metric=BinaryBinnedAUPRC(num_tasks=num_tasks, threshold=threshold),
            state_names={"inputs", "targets"},
            update_kwargs={
                "input": input,
                "target": target,
            },
            compute_result=compute_result,
        )

    def test_auprc_class_valid_input(self) -> None:
        torch.manual_seed(123)
        # test case with num_tasks=1
        input = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        threshold = 5
        self._test_auprc_class_with_input(
            input,
            target,
            num_tasks=1,
            threshold=threshold,
            compute_result=(
                torch.tensor(0.5117788314819336),
                torch.tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]),
            ),
        )

        # test case with num_tasks=2
        torch.manual_seed(123)
        num_tasks = 2
        input = torch.rand(NUM_TOTAL_UPDATES, num_tasks, BATCH_SIZE)
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, num_tasks, BATCH_SIZE))
        threshold = 5
        self._test_auprc_class_with_input(
            input,
            target,
            num_tasks=num_tasks,
            threshold=threshold,
            compute_result=(
                torch.tensor(
                    [0.5810506343841553, 0.5106710195541382],
                ),
                torch.tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]),
            ),
        )

        # test case with different update shape
        num_classes = 2
        threshold = 5
        torch.manual_seed(123)
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
        compute_result = (
            torch.tensor(0.42704516649246216),
            torch.tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]),
        )

        self.run_class_implementation_tests(
            metric=BinaryBinnedAUPRC(threshold=threshold),
            state_names={"inputs", "targets"},
            update_kwargs={
                "input": update_input,
                "target": update_target,
            },
            compute_result=compute_result,
            num_total_updates=4,
            num_processes=2,
        )

    def test_binary_binned_auprc_class_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks` has to be at least 1.",
        ):
            BinaryBinnedAUPRC(num_tasks=-1)

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same shape, "
            r"got shapes torch.Size\(\[4\]\) and torch.Size\(\[3\]\).",
        ):
            metric = BinaryBinnedAUPRC()
            metric.update(torch.rand(4), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks = 1`, `input` is expected to be 1D or 2D tensor, but got shape "
            r"torch.Size\(\[\]\).",
        ):
            metric = BinaryBinnedAUPRC()
            metric.update(torch.rand(size=()), torch.rand(size=()))

        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks = 1`, `input` is expected to be 1D or 2D tensor, but got shape "
            r"torch.Size\(\[4, 5, 5\]\).",
        ):
            metric = BinaryBinnedAUPRC()
            metric.update(torch.rand(4, 5, 5), torch.rand(4, 5, 5))

        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks = 2`, `input` is expected to be 2D tensor, but got shape "
            r"torch.Size\(\[4, 5, 5\]\).",
        ):
            metric = BinaryBinnedAUPRC(num_tasks=2)
            metric.update(torch.rand(4, 5, 5), torch.rand(4, 5, 5))

        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks = 2`, `input`'s shape is expected to be "
            r"\(2, num_samples\), but got shape torch.Size\(\[4, 5\]\).",
        ):
            metric = BinaryBinnedAUPRC(num_tasks=2)
            metric.update(torch.rand(4, 5), torch.rand(4, 5))

        with self.assertRaisesRegex(
            ValueError, "The `threshold` should be a sorted tensor."
        ):
            metric = BinaryBinnedAUPRC(
                threshold=torch.tensor([0.1, 0.2, 0.5, 0.7, 0.6]),
            )

        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            metric = BinaryBinnedAUPRC(
                threshold=torch.tensor([-0.1, 0.2, 0.5, 0.7]),
            )

        with self.assertRaisesRegex(
            ValueError,
            r"`threshold` should be 1-dimensional, but got 2D tensor.",
        ):
            metric = BinaryBinnedAUPRC(
                threshold=torch.tensor([[-0.1, 0.2, 0.5, 0.7], [0.0, 0.4, 0.6, 1.0]]),
            )

        with self.assertRaisesRegex(
            ValueError,
            r"First value in `threshold` should be 0.",
        ):
            metric = BinaryBinnedAUPRC(
                threshold=torch.tensor([0.1, 0.2, 0.5, 1.0]),
            )

        with self.assertRaisesRegex(
            ValueError,
            r"Last value in `threshold` should be 1.",
        ):
            metric = BinaryBinnedAUPRC(
                threshold=torch.tensor([0.0, 0.2, 0.5, 0.9]),
            )
