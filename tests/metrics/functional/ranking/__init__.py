# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from tests.metrics.functional.ranking.test_hit_rate import TestHitRate
from tests.metrics.functional.ranking.test_reciprocal_rank import TestReciprocalRank

__all__ = [
    "TestHitRate",
    "TestReciprocalRank",
]
