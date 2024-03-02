# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from torcheval.metrics.aggregation.auc import AUC
from torcheval.metrics.aggregation.cat import Cat
from torcheval.metrics.aggregation.max import Max
from torcheval.metrics.aggregation.mean import Mean
from torcheval.metrics.aggregation.min import Min
from torcheval.metrics.aggregation.sum import Sum
from torcheval.metrics.aggregation.throughput import Throughput

__all__ = ["AUC", "Cat", "Max", "Mean", "Min", "Sum", "Throughput"]
__doc_name__ = "Aggregation Metrics"
