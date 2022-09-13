# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]: Pyre was not able to infer the type of argument

import unittest
from typing import Optional

import torch

from sklearn.metrics import roc_auc_score

from torcheval.metrics import BinaryAUROC
from torcheval.utils.test_utils.metric_class_tester import (
    BATCH_SIZE,
    MetricClassTester,
    NUM_TOTAL_UPDATES,
)


class TestBinaryAUROC(MetricClassTester):
    def _test_auroc_class_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_tasks: int = 1,
        compute_result: Optional[torch.Tensor] = None,
        use_fbgemm: Optional[bool] = False,
    ) -> None:
        input_tensors = input.reshape(-1, 1)
        target_tensors = target.reshape(-1, 1)
        if compute_result is None:
            compute_result = torch.tensor(roc_auc_score(target_tensors, input_tensors))

        self.run_class_implementation_tests(
            metric=BinaryAUROC(num_tasks=num_tasks, use_fbgemm=use_fbgemm),
            state_names={"inputs", "targets"},
            update_kwargs={
                "input": input,
                "target": target,
            },
            compute_result=compute_result,
            test_devices=["cuda"] if use_fbgemm else None,
        )

    def _test_auroc_class_set(self, use_fbgemm: Optional[bool] = False) -> None:
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        self._test_auroc_class_with_input(input, target, use_fbgemm=use_fbgemm)

        if not use_fbgemm:
            # Skip this test for use_fbgemm because FBGEMM AUC is an
            # approximation of AUC. It can give a significantly different
            # result if input data is highly redundant
            input = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
            target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
            self._test_auroc_class_with_input(
                input,
                target,
                use_fbgemm=use_fbgemm,
            )

        num_tasks = 2
        torch.manual_seed(123)
        input = torch.rand(NUM_TOTAL_UPDATES, num_tasks, BATCH_SIZE)
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, num_tasks, BATCH_SIZE))
        self._test_auroc_class_with_input(
            input,
            target,
            num_tasks=2,
            compute_result=torch.tensor(
                [0.549048678033111, 0.512218963831867], dtype=torch.float64
            ),
            use_fbgemm=use_fbgemm,
        )

    @unittest.skipUnless(
        condition=torch.cuda.is_available(), reason="This test needs a GPU host to run."
    )
    def test_auroc_class_fbgemm(self) -> None:
        self._test_auroc_class_set(use_fbgemm=True)

    def test_auroc_class_base(self) -> None:
        self._test_auroc_class_set(use_fbgemm=False)

    def test_auroc_class_update_input_shape_different(self) -> None:
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
        compute_result = torch.tensor(
            roc_auc_score(
                torch.cat(update_target, dim=0),
                torch.cat(update_input, dim=0),
            ),
        )

        self.run_class_implementation_tests(
            metric=BinaryAUROC(),
            state_names={"inputs", "targets"},
            update_kwargs={
                "input": update_input,
                "target": update_target,
            },
            compute_result=compute_result,
            num_total_updates=4,
            num_processes=2,
        )

    def test_auroc_class_invalid_input(self) -> None:
        metric = BinaryAUROC()
        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same shape, "
            r"got shapes torch.Size\(\[4\]\) and torch.Size\(\[3\]\).",
        ):
            metric.update(torch.rand(4), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks = 1`, `input` is expected to be one-dimensional tensor,",
        ):
            metric.update(torch.rand(4, 5), torch.rand(4, 5))

        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks = 2`, `input`'s shape is expected to be",
        ):
            metric = BinaryAUROC(num_tasks=2)
            metric.update(torch.rand(4, 5), torch.rand(4, 5))

        with self.assertRaisesRegex(ValueError, "`num_tasks` value should be greater"):
            metric = BinaryAUROC(num_tasks=0)
