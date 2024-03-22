# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[16]: Undefined attribute of metric states.

# pyre-fixme[21]: Could not find module
#  `torcheval.metrics.functional.regression.mean_squared_error`.
from torcheval.metrics.functional.regression.mean_squared_error import (
    mean_squared_error,
)

# pyre-fixme[21]: Could not find module
#  `torcheval.metrics.functional.regression.r2_score`.
from torcheval.metrics.functional.regression.r2_score import r2_score

__all__ = ["mean_squared_error", "r2_score"]
__doc_name__ = "Regression Metrics"
