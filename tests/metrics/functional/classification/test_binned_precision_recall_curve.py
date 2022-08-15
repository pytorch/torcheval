# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Tuple, Union

import torch
from torcheval.metrics.functional import binary_binned_precision_recall_curve


class TestBinaryBinnedPrecisionRecallCurve(unittest.TestCase):
    def _test_binary_binned_precision_recall_curve_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        threshold: Union[torch.Tensor, int],
        compute_result: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        my_compute_result = binary_binned_precision_recall_curve(
            input,
            target,
            threshold=threshold,
        )
        _test_helper(input, target, my_compute_result, compute_result)

    def test_binary_binned_precision_recall_curve_base(self) -> None:
        input = torch.tensor([0.2, 0.3, 0.4, 0.5])
        target = torch.tensor([0, 0, 1, 1])
        threshold = torch.tensor([0.0000, 0.2500, 0.7500, 1.0000])

        compute_result = (
            torch.tensor([0.5000, 2 / 3, 1.0000, 1.0000, 1.0000]),
            torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0000, 0.2500, 0.7500, 1.0000]),
        )
        self._test_binary_binned_precision_recall_curve_with_input(
            input, target, threshold, compute_result
        )

        if torch.cuda.is_available():
            self._test_binary_binned_precision_recall_curve_with_input(
                input.to("cuda"),
                target.to("cuda"),
                threshold.to("cuda"),
                tuple([c.to("cuda") for c in compute_result]),
            )

        input = torch.tensor([0.2, 0.8, 0.5, 0.9])
        target = torch.tensor([0, 1, 0, 1])
        threshold = 5
        compute_result = (
            torch.tensor(
                [
                    0.5000,
                    2 / 3,
                    2 / 3,
                    1.0000,
                    1.0000,
                    1.0000,
                ]
            ),
            torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
            torch.tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]),
        )
        self._test_binary_binned_precision_recall_curve_with_input(
            input, target, threshold, compute_result
        )

        if torch.cuda.is_available():
            self._test_binary_binned_precision_recall_curve_with_input(
                input.to("cuda"),
                target.to("cuda"),
                threshold,
                tuple([c.to("cuda") for c in compute_result]),
            )

    def test_binary_binned_precision_recall_curve_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "input should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            binary_binned_precision_recall_curve(torch.rand(3, 2), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "target should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            binary_binned_precision_recall_curve(torch.rand(3), torch.rand(3, 2))

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same shape, "
            r"got shapes torch.Size\(\[4\]\) and torch.Size\(\[3\]\).",
        ):
            binary_binned_precision_recall_curve(torch.rand(4), torch.rand(3))
        with self.assertRaisesRegex(
            ValueError, "The `threshold` should be a sorted array."
        ):
            binary_binned_precision_recall_curve(
                torch.rand(4),
                torch.rand(4),
                threshold=torch.tensor([0.1, 0.2, 0.5, 0.7, 0.6]),
            )
        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            binary_binned_precision_recall_curve(
                torch.rand(4),
                torch.rand(4),
                threshold=torch.tensor([-0.1, 0.2, 0.5, 0.7]),
            )
        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            binary_binned_precision_recall_curve(
                torch.rand(4),
                torch.rand(4),
                threshold=torch.tensor([0.1, 0.2, 0.5, 1.7]),
            )


def _test_helper(
    input: torch.Tensor,
    target: torch.Tensor,
    my_compute_result: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    compute_result: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    for my_tensor, tensor in zip(my_compute_result, compute_result):
        torch.testing.assert_close(
            my_tensor,
            tensor,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )
