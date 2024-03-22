# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-fixme[21]: Could not find module
#  `torcheval.metrics.functional.ranking.click_through_rate`.
from torcheval.metrics.functional.ranking.click_through_rate import click_through_rate
from torcheval.metrics.functional.ranking.frequency import frequency_at_k

# pyre-fixme[21]: Could not find module `torcheval.metrics.functional.ranking.hit_rate`.
from torcheval.metrics.functional.ranking.hit_rate import hit_rate

# pyre-fixme[21]: Could not find module
#  `torcheval.metrics.functional.ranking.num_collisions`.
from torcheval.metrics.functional.ranking.num_collisions import num_collisions

# pyre-fixme[21]: Could not find module
#  `torcheval.metrics.functional.ranking.reciprocal_rank`.
from torcheval.metrics.functional.ranking.reciprocal_rank import reciprocal_rank

# pyre-fixme[21]: Could not find module
#  `torcheval.metrics.functional.ranking.retrieval_precision`.
from torcheval.metrics.functional.ranking.retrieval_precision import retrieval_precision

# pyre-fixme[21]: Could not find module
#  `torcheval.metrics.functional.ranking.retrieval_recall`.
from torcheval.metrics.functional.ranking.retrieval_recall import retrieval_recall

# pyre-fixme[21]: Could not find module
#  `torcheval.metrics.functional.ranking.weighted_calibration`.
from torcheval.metrics.functional.ranking.weighted_calibration import (
    weighted_calibration,
)

__all__ = [
    "click_through_rate",
    "frequency_at_k",
    "hit_rate",
    "num_collisions",
    "reciprocal_rank",
    "weighted_calibration",
    "retrieval_precision",
    "retrieval_recall",
]
__doc_name__ = "Ranking Metrics"
