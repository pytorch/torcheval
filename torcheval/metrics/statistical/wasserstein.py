import logging
from typing import Iterable, Optional, TypeVar, Union

import torch

from torcheval.metrics.functional.statistical.wasserstein import (
    _wasserstein_param_check,
    _wasserstein_update_input_check,
    _wasserstein_compute
)
from torcheval.metrics.metric import Metric

TWasserstein = TypeVar("TWasserstein")

class Wasserstein1D(Metric[torch.Tensor]):

    def __init__(
        self: TWasserstein,
        *,
        device: Optional[torch.device] = None
    ) -> None:
        super().__init__(device=device)
        # Keeping record of samples
        self._add_state("dist_1_samples",
                        torch.Tensor([], device = self.device)
        )
        self._add_state("dist_2_samples",
                        torch.Tensor([], device = self.device)
        )
        self._add_state("dist_1_weights",
                        torch.Tensor([], device = self.device)
        )
        self._add_state("dist_2_weights",
                        torch.Tensor([], device = self.device)
        )

    @torch.inference_mode()
    def update(self,
               new_samples_dist_1: torch.Tensor,
               new_samples_dist_2: torch.Tensor,
               new_weights_dist_1: Optional[torch.Tensor]=None,
               new_weights_dist_2: Optional[torch.Tensor]=None
    ) -> None:
        new_samples_dist_1 = new_samples_dist_1.to(self.device)
        new_samples_dist_2 = new_samples_dist_2.to(self.device)
        new_weights_dist_1 = new_weights_dist_1.to(self.device)
        new_weights_dist_2 = new_weights_dist_2.to(self.device)

        _wasserstein_param_check(new_samples_dist_1, new_weights_dist_1)
        _wasserstein_param_check(new_samples_dist_2, new_weights_dist_2)

        _wasserstein_update_input_check(new_samples_dist_1, new_weights_dist_1)
        _wasserstein_update_input_check(new_samples_dist_2, new_weights_dist_2)

        # When new data comes in, just add them to the list of samples
        self.dist_1_samples = torch.cat((self.dist_1_samples, new_samples_dist_1))
        self.dist_2_samples = torch.cat((self.dist_2_samples, new_samples_dist_2))
        self.dist_1_weights = torch.cat((self.dist_1_weights, new_weights_dist_1))
        self.dist_2_weights = torch.cat((self.dist_2_weights, new_weights_dist_2))

        return self

    @torch.inference_mode()
    def compute(self):
        return _wasserstein_compute(self.dist_1_samples, self.dist_2_samples,
                                    self.dist_1_weights, self.dist_2_weights
        )

    @torch.inference_mode()
    def merge_state(
        self: TWasserstein,
        metrics: Iterable[TWasserstein]
    ) -> TWasserstein:
        # Concatenating all the samples for each distribution
        dist_1_samples = [self.dist_1_samples, ]
        dist_2_samples = [self.dist_1_samples, ]
        dist_1_weights = [self.dist_1_weights, ]
        dist_2_weights = [self.dist_2_weights, ]

        for metric in metrics:
            dist_1_samples.append(metric.dist_1_samples)
            dist_2_samples.append(metric.dist_2_samples)
            dist_1_weights.append(metric.dist_1_weights)
            dist_2_weights.append(metric.dist_2_weights)

        self.dist_1_samples = torch.cat(dist_1_samples)
        self.dist_2_samples = torch.cat(dist_2_samples)
        self.dist_1_weights = torch.cat(dist_1_weights)
        self.dist_2_weights = torch.cat(dist_2_weights)

        return self