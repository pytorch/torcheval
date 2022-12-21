# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torcheval.metrics import functional
from torcheval.metrics.aggregation import AUC, Cat, Max, Mean, Min, Sum, Throughput
from torcheval.metrics.classification import (
    BinaryAccuracy,
    BinaryAUPRC,
    BinaryAUROC,
    BinaryBinnedAUROC,
    BinaryBinnedPrecisionRecallCurve,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryNormalizedEntropy,
    BinaryPrecision,
    BinaryPrecisionRecallCurve,
    BinaryRecall,
    BinaryRecallAtFixedPrecision,
    MulticlassAccuracy,
    MulticlassAUPRC,
    MulticlassAUROC,
    MulticlassBinnedAUROC,
    MulticlassBinnedPrecisionRecallCurve,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassPrecisionRecallCurve,
    MulticlassRecall,
    MultilabelAccuracy,
    MultilabelAUPRC,
    MultilabelPrecisionRecallCurve,
    MultilabelRecallAtFixedPrecision,
    TopKMultilabelAccuracy,
)
from torcheval.metrics.metric import Metric

from torcheval.metrics.ranking import (
    ClickThroughRate,
    HitRate,
    ReciprocalRank,
    WeightedCalibration,
)
from torcheval.metrics.regression import MeanSquaredError, R2Score

from torcheval.metrics.text import (
    Perplexity,
    WordErrorRate,
    WordInformationLost,
    WordInformationPreserved,
)
from torcheval.metrics.window import (
    WindowedBinaryAUROC,
    WindowedBinaryNormalizedEntropy,
    WindowedClickThroughRate,
    WindowedMeanSquaredError,
    WindowedWeightedCalibration,
)

__all__ = [
    # base interface
    "Metric",
    # functional metrics
    "functional",
    # class metrics
    "AUC",
    "BinaryAUPRC",
    "BinaryAUROC",
    "BinaryAccuracy",
    "BinaryBinnedAUROC",
    "BinaryBinnedPrecisionRecallCurve",
    "BinaryConfusionMatrix",
    "BinaryF1Score",
    "BinaryNormalizedEntropy",
    "BinaryPrecision",
    "BinaryPrecisionRecallCurve",
    "BinaryRecall",
    "BinaryRecallAtFixedPrecision",
    "Cat",
    "ClickThroughRate",
    "HitRate",
    "Max",
    "Mean",
    "MeanSquaredError",
    "Min",
    "MulticlassAccuracy",
    "MulticlassAUPRC",
    "MulticlassAUROC",
    "MulticlassBinnedAUROC",
    "MulticlassBinnedPrecisionRecallCurve",
    "MulticlassConfusionMatrix",
    "MulticlassF1Score",
    "MulticlassPrecision",
    "MulticlassPrecisionRecallCurve",
    "MulticlassRecall",
    "MultilabelAccuracy",
    "MultilabelAUPRC",
    "MultilabelPrecisionRecallCurve",
    "MultilabelRecallAtFixedPrecision",
    "Perplexity",
    "TopKMultilabelAccuracy",
    "R2Score",
    "ReciprocalRank",
    "Sum",
    "Throughput",
    "WeightedCalibration",
    "WindowedBinaryAUROC",
    "WindowedBinaryNormalizedEntropy",
    "WindowedClickThroughRate",
    "WindowedMeanSquaredError",
    "WindowedWeightedCalibration",
    "WordErrorRate",
    "WordInformationPreserved",
    "WordInformationLost",
]
