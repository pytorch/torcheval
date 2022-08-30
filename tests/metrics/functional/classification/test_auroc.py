# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Optional

import torch
from sklearn.metrics import roc_auc_score
from torcheval.metrics.functional import binary_auroc
from torcheval.utils.test_utils.metric_class_tester import BATCH_SIZE


class TestBinaryAUROC(unittest.TestCase):
    def _test_auroc_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_tasks: int = 1,
        compute_result: Optional[torch.Tensor] = None,
    ) -> None:
        if compute_result is None:
            compute_result = torch.tensor(roc_auc_score(target, input))
        if torch.cuda.is_available():
            my_compute_result = binary_auroc(
                input.to(device="cuda"), target.to(device="cuda"), num_tasks=num_tasks
            )
            compute_result = compute_result.to(device="cuda")
        else:
            my_compute_result = binary_auroc(input, target, num_tasks=num_tasks)

        torch.testing.assert_close(
            my_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_auroc_base(self) -> None:
        input = torch.tensor([1, 1, 0, 0])
        target = torch.tensor([1, 0, 1, 0])
        self._test_auroc_with_input(input, target)

        input = torch.rand(BATCH_SIZE)
        target = torch.randint(high=2, size=(BATCH_SIZE,))
        self._test_auroc_with_input(input, target)

        input = torch.tensor([[1, 1, 1, 0], [0.1, 0.5, 0.7, 0.8]])
        target = torch.tensor([[1, 0, 1, 0], [1, 0, 1, 1]])
        self._test_auroc_with_input(
            input,
            target,
            2,
            torch.tensor([0.7500, 0.6666666666666666], dtype=torch.float64),
        )

    def test_auroc_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same shape, "
            r"got shapes torch.Size\(\[4\]\) and torch.Size\(\[3\]\).",
        ):
            binary_auroc(torch.rand(4), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks = 1`, `input` is expected to be one-dimensional tensor,",
        ):
            binary_auroc(torch.rand(4, 5), torch.rand(4, 5))

        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks = 2`, `input`'s shape is expected to be",
        ):
            binary_auroc(torch.rand(4, 5), torch.rand(4, 5), num_tasks=2)
