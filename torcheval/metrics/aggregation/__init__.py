# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torcheval.metrics.aggregation.cat import Cat
from torcheval.metrics.aggregation.max import Max
from torcheval.metrics.aggregation.mean import Mean
from torcheval.metrics.aggregation.min import Min
from torcheval.metrics.aggregation.sum import Sum

__all__ = ["Cat", "Max", "Mean", "Min", "Sum"]
