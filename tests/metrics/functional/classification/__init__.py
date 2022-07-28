# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from tests.metrics.functional.classification.test_accuracy import TestAccuracy
from tests.metrics.functional.classification.test_auroc import TestAUROC
from tests.metrics.functional.classification.test_f1_score import TestF1Score
from tests.metrics.functional.classification.test_multi_label_accuracy import (
    TestMultiLabelAccuracy,
)
from tests.metrics.functional.classification.test_precision import TestPrecision
from tests.metrics.functional.classification.test_precision_recall_curve import (
    TestBinaryPrecisionRecallCurve,
    TestMulticlassPrecisionRecallCurve,
)
from tests.metrics.functional.classification.test_recall import TestRecall

__all__ = [
    "TestAccuracy",
    "TestAUROC",
    "TestBinaryPrecisionRecallCurve",
    "TestF1Score",
    "TestMultiLabelAccuracy",
    "TestMulticlassPrecisionRecallCurve",
    "TestPrecision",
    "TestRecall",
]
