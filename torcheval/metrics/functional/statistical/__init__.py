# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[16]: Undefined attribute of metric states.

from torcheval.metrics.functional.statistical.wasserstein import wasserstein_1d

__all__ = ["wasserstein_1d"]
__doc_name__ = "Statistical Metrics"
