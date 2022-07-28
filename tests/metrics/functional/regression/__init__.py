# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[16]: Undefined attribute of metric states.

from tests.metrics.functional.regression.test_mean_squared_error import (
    TestMeanSquaredError,
)
from tests.metrics.functional.regression.test_r2_score import TestR2Score

__all__ = ["TestMeanSquaredError", "TestR2Score"]
