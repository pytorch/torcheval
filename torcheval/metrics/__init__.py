# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torcheval.metrics import functional
from torcheval.metrics.aggregation import Cat, Max, Mean, Min, Sum, Throughput
from torcheval.metrics.classification import (
    Accuracy,
    AUROC,
    F1Score,
    MultiLabelAccuracy,
    Recall,
)
from torcheval.metrics.metric import Metric
from torcheval.metrics.ranking import ReciprocalRank
from torcheval.metrics.regression import MeanSquaredError, R2Score

__all__ = [
    ## base interface
    "Metric",
    ## functional metrics
    "functional",
    ## class metrics
    "Accuracy",
    "AUROC",
    "Cat",
    "F1Score",
    "MultiLabelAccuracy",
    "Precision",
    "Recall",
    "Max",
    "Mean",
    "MeanSquaredError",
    "Min",
    "R2Score",
    "ReciprocalRank",
    "Sum",
    "Throughput",
]
