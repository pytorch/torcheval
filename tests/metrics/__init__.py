# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from tests.metrics import functional
from tests.metrics.aggregation import TestCat, TestMax, TestMean, TestMin, TestSum, TestThroughput
from tests.metrics.classification import (
    TestAccuracy,
    TestAUROC,
    TestBinaryPrecisionRecallCurve,
    TestF1Score,
    TestMulticlassPrecisionRecallCurve,
    TestMultiLabelAccuracy,
    TestRecall,
)
from tests.metrics.test_metric import MetricBaseClassTest
from tests.metrics.ranking import TestReciprocalRank
from tests.metrics.regression import TestMeanSquaredError, TestR2Score

__all__ = [
    ## base interface
    "MetricBaseClassTest",
    ## functional metrics
    "functional",
    ## class metrics
    "TestAccuracy",
    "TestAUROC",
    "TestBinaryPrecisionRecallCurve",
    "TestCat",
    "TestF1Score",
    "TestMultiLabelAccuracy",
    "TestMulticlassPrecisionRecallCurve",
    "TestPrecision",
    "TestRecall",
    "TestMax",
    "TestMean",
    "TestMeanSquaredError",
    "TestMin",
    "TestR2Score",
    "TestReciprocalRank",
    "TestSum",
    "TestThroughput",
]
