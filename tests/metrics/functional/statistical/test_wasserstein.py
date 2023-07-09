# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import List, Optional, Tuple, Union

import torch
from torcheval.metrics.functional.statistical.wasserstein import wasserstein_1d
from torcheval.utils import random_data as rd

class TestWasserstein1D(unittest.TestCase):
    def _test_wasserstein1d_with_input(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            x_weights: Optional[torch.Tensor],
            y_weights: Optional[torch.Tensor],
            compute_result: torch.Tensor,
    ) -> None:
        my_compute_result = wasserstein_1d(x, y, x_weights, y_weights)

        torch.testing.assert_allclose(
            my_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8
            rtol=1e-5,
        )


    def _test_with_randomized_data_getter(self) -> None:
        pass
