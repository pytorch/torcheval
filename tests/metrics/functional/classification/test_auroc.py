# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from sklearn.metrics import roc_auc_score
from torcheval.metrics.functional import binary_auroc
from torcheval.utils.test_utils.metric_class_tester import BATCH_SIZE


class TestBinaryAUROC(unittest.TestCase):
    def _test_auroc_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        compute_result = torch.tensor(roc_auc_score(target, input))
        if torch.cuda.is_available():
            my_compute_result = binary_auroc(
                input.to(device="cuda"), target.to(device="cuda")
            )
            compute_result = compute_result.to(device="cuda")
        else:
            my_compute_result = binary_auroc(input, target)

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

        input = torch.rand(BATCH_SIZE)
        target = torch.randint(high=2, size=(BATCH_SIZE,))
        self._test_auroc_with_input(input, target)

    def test_auroc_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "input should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            binary_auroc(torch.rand(3, 2), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "target should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            binary_auroc(torch.rand(3), torch.rand(3, 2))

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same shape, "
            r"got shapes torch.Size\(\[4\]\) and torch.Size\(\[3\]\).",
        ):
            binary_auroc(torch.rand(4), torch.rand(3))
