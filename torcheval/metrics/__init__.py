# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torcheval.metrics import functional
from torcheval.metrics.aggregation import Cat, Max, Mean, Min, Sum, Throughput
from torcheval.metrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryBinnedPrecisionRecallCurve,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryNormalizedEntropy,
    BinaryPrecision,
    BinaryPrecisionRecallCurve,
    BinaryRecall,
    MulticlassAccuracy,
    MulticlassBinnedPrecisionRecallCurve,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassPrecisionRecallCurve,
    MulticlassRecall,
    MultilabelAccuracy,
    TopKMultilabelAccuracy,
)
from torcheval.metrics.metric import Metric
from torcheval.metrics.ranking import HitRate, ReciprocalRank, WeightedCalibration
from torcheval.metrics.regression import MeanSquaredError, R2Score
from torcheval.metrics.window import WindowedBinaryNormalizedEntropy

__all__ = [
    # base interface
    "Metric",
    # functional metrics
    "functional",
    # class metrics
    "BinaryAUROC",
    "BinaryAccuracy",
    "BinaryBinnedPrecisionRecallCurve",
    "BinaryConfusionMatrix",
    "BinaryF1Score",
    "BinaryNormalizedEntropy",
    "BinaryPrecision",
    "BinaryPrecisionRecallCurve",
    "BinaryRecall",
    "Cat",
    "HitRate",
    "Max",
    "Mean",
    "MeanSquaredError",
    "Min",
    "MulticlassAccuracy",
    "MulticlassBinnedPrecisionRecallCurve",
    "MulticlassConfusionMatrix",
    "MulticlassF1Score",
    "MulticlassPrecision",
    "MulticlassPrecisionRecallCurve",
    "MulticlassRecall",
    "MultilabelAccuracy",
    "TopKMultilabelAccuracy",
    "R2Score",
    "ReciprocalRank",
    "Sum",
    "Throughput",
    "WeightedCalibration",
    "WindowedBinaryNormalizedEntropy",
]
