# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import List, Optional, Tuple, Union

from scipy.stats import wasserstein_distance as sp_wasserstein
import numpy as np

import torch
from torcheval.metrics.functional.statistical.wasserstein import wasserstein_1d
from torcheval.utils import random_data as rd

class TestWasserstein1D(unittest.TestCase):
    def _get_scipy_equivalent(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_weights: Optional[torch.Tensor]=None,
        y_weights: Optional[torch.Tensor]=None,
        device: str="cpu"
    ) -> torch.Tensor:

        # Convert inputs to scipy style inputs
        sp_x = x.numpy()
        sp_y = y.numpy()
        shape = x.shape

        if x_weights is not None and y_weights is not None:
            sp_x_weights = x_weights.numpy()
            sp_y_weights = y_weights.numpy()
            sp_wd = []
            if len(shape) > 1:
                for i in range(sp_x.shape[0]):
                    sp_x_i = sp_x[i, :]
                    sp_y_i = sp_y[i, :]
                    sp_x_w_i = sp_x_weights[i, :]
                    sp_y_w_i = sp_y_weights[i, :]
                    sp_wd.append(np.nan_to_num(sp_wasserstein(sp_x_i, sp_y_i, sp_x_w_i, sp_y_w_i)))
            else:
                sp_wd.append(np.nan_to_num(sp_wasserstein(sp_x, sp_y, sp_x_weights, sp_y_weights)))
            return torch.tensor(sp_wd, device=device).to(torch.float32)
        else:
            if len(shape) > 1:
                for i in range(sp_x.shape[0]):
                    sp_x_i = sp_x[i, :]
                    sp_y_i = sp_y[i, :]
                    sp_wd.append(np.nan_to_num(sp_wasserstein(sp_x_i, sp_y_i)))
            else:
                sp_wd.append(np.nan_to_num(sp_wasserstein(sp_x, sp_y, sp_x_weights, sp_y_weights)))
        return torch.tensor(sp_wd, device=device).to(torch.float32)
    

    def _test_wasserstein1d_with_input(
        self,
        compute_result: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        x_weights: Optional[torch.Tensor]=None,
        y_weights: Optional[torch.Tensor]=None
    ) -> None:
        my_compute_result = wasserstein_1d(x, y, x_weights, y_weights)
        torch.testing.assert_allclose(
            my_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

        ## TODO: Add CUDA check

    def test_wasserstein1d_distribution_values_only(self) -> None:
        x = torch.Tensor([5, -5, -7, 9, -3])
        y = torch.Tensor([9, -7, 5, -4, -2])
        self._test_wasserstein1d_with_input(
            torch.Tensor([0.39999999999999997]),
            x, y
        )

    def test_wasserstein1d_distribution_and_weight_values(self) -> None:
        x = torch.Tensor([-13, -9, -19, 11, -18, -20, 8, 2, -8, -18])
        y = torch.Tensor([9, 6, -5, -11, 9, -4, -13, -19, -14, 4])
        x_weights = torch.Tensor([3, 3, 1, 2, 2, 3, 2, 2, 2, 3])
        y_weights = torch.Tensor([2, 2, 1, 1, 2, 2, 1, 1, 1, 1])
        self._test_wasserstein1d_with_input(
            torch.Tensor([8.149068322981368]),
            x, y,
            x_weights, y_weights
        )

    def test_wasserstein1d_identical_distributions(self) -> None:
        x = torch.Tensor([-13, -9, -19, 11, -18, -20, 8, 2, -8, -18])
        x_weights = torch.Tensor([3, 3, 1, 2, 2, 3, 2, 2, 2, 3])
        self._test_wasserstein1d_with_input(
            torch.Tensor([0.0]),
            x, x,
            x_weights, x_weights
        )

    def test_wasserstein1d_randomized_data_getter(self) -> None:
        num_updates = 1
        batch_size = 32
        device = "cuda" if torch.cuda.is_available() else "cpu"

        for _ in range(100):
            x, y, x_weights, y_weights = rd.get_rand_data_wasserstein1d(
                num_updates,
                batch_size,
                device
            )

            compute_result = self._get_scipy_equivalent(x, y, x_weights, y_weights, device)

            self._test_wasserstein1d_with_input(
                compute_result,
                x,
                y,
                x_weights,
                y_weights)


    def test_wasserstein1d_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "The two distributions should have same shape, "
            "got shapes "
            r"torch.Size\(\[4\]\) and torch.Size\(\[7\]\)."
        ):
            wasserstein_1d(torch.rand(4), torch.rand(7))
        
        with self.assertRaisesRegex(
            ValueError,
            "Distribution cannot be empty."
        ):
            wasserstein_1d(torch.rand(4), torch.Tensor([]))
        
        with self.assertRaisesRegex(
            ValueError,
            "Distribution cannot be empty."
        ):
            wasserstein_1d(torch.Tensor([]), torch.rand(5))

        with self.assertRaisesRegex(
            ValueError,
            "Weight tensor sum must be positive-finite."
        ):
            wasserstein_1d(torch.rand(4), torch.rand(4),
                           torch.Tensor([]), torch.rand(4)
            )

        with self.assertRaisesRegex(
            ValueError,
            "Weight tensor sum must be positive-finite."
        ):
            wasserstein_1d(torch.rand(4), torch.rand(4),
                           torch.rand(4), torch.Tensor([])
            )
        
        with self.assertRaisesRegex(
            ValueError,
            "Distribution values and weight tensors must be of the same shape, "
            "got shapes "
            r"torch.Size\(\[4\]\) and torch.Size\(\[7\]\)."
        ):
            wasserstein_1d(torch.rand(4), torch.rand(4),
                           torch.rand(7), torch.rand(4)
            )

        with self.assertRaisesRegex(
            ValueError,
            "Distribution values and weight tensors must be of the same shape, "
            "got shapes "
            r"torch.Size\(\[6\]\) and torch.Size\(\[10\]\)."
        ):
            wasserstein_1d(torch.rand(6), torch.rand(6),
                           torch.rand(6), torch.rand(10)
            )
        
        with self.assertRaisesRegex(
            ValueError,
            "All weights must be non-negative."
        ):
            wasserstein_1d(torch.rand(4), torch.rand(4),
                           torch.Tensor([1, -1, 2, 3]), torch.rand(4)
            )

        with self.assertRaisesRegex(
            ValueError,
            "All weights must be non-negative."
        ):
            wasserstein_1d(torch.rand(4), torch.rand(4),
                           torch.rand(4), torch.Tensor([1, -1, 2, 3])
            )

        with self.assertRaisesRegex(
            ValueError,
            "All weights must be non-negative."
        ):
            wasserstein_1d(torch.rand(4), torch.rand(4),
                           torch.Tensor([-1.0, -2.0, 0.0, 1.0]), torch.rand(4)
            )

        with self.assertRaisesRegex(
            ValueError,
            "All weights must be non-negative."
        ):
            wasserstein_1d(torch.rand(4), torch.rand(4),
                           torch.rand(4), torch.Tensor([-1.5, -1.0, 0.5, 0.75])
            )
