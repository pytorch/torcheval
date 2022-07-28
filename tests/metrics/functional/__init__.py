# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from tests.metrics.functional.aggregation import TestMean, TestSum
from tests.metrics.functional.classification import (
    TestAccuracy,
    TestAUROC,
    TestBinaryPrecisionRecallCurve,
    TestF1Score,
    TestMulticlassPrecisionRecallCurve,
    TestMultiLabelAccuracy,
    TestPrecision,
    TestRecall,
)
from tests.metrics.functional.ranking import TestHitRate, TestReciprocalRank
from tests.metrics.functional.regression import TestMeanSquaredError, TestR2Score

__all__ = [
    "TestAccuracy",
    "TestAUROC",
    "TestBinaryPrecisionRecallCurve",
    "TestF1Score",
    "TestHitRate",
    "TestPrecision",
    "TestMean",
    "TestMultiLabelAccuracy",
    "TestMeanSquaredError",
    "TestMulticlassPrecisionRecallCurve",
    "TestR2Score",
    "TestReciprocalRank",
    "TestSum",
    "TestRecall",
]
