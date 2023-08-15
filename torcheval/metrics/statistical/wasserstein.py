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
    """
    The Wasserstein distance, also called the Earth Mover's Distance, is a 
    measure of the similarity between two distributions.

    The Wasserstein distance between two distributions is intuitively the 
    minimum weight of soil (times distance moved) that would need to be moved 
    if the two distributions were represented by two piles of soil.

    Its functional version is :func:'torcheval.metrics.functional.statistical.wasserstein'.

    Examples
    --------
    >>> from torcheval.metrics import Wasserstein1D
    >>> metric = Wasserstein1D()
    >>> metric.update(torch.tensor([0,1,2,2]), torch.tensor([0,1]))
    >>> metric.compute()
    0.75
    >>> metric = Wasserstein1D()
    >>> metric.update(torch.tensor([0,1,2]), torch.tensor([0,1,1]), torch.tensor([1,2,0]), torch.tensor([1,1,1]))
    >>> metric.compute()
    0
    >>> metric = Wasserstein1D()
    >>> metric.update(torch.tensor([0,1,2]), torch.tensor([0,1,1]))
    >>> metric.compute()
    0.33333333333333337
    >>> metric.update(torch.tensor([1,1,1]), torch.tensor([1,1,1]))
    >>> metric.compute()
    0.16666666666666663
    """
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
        self._add_state("dist_1_weights",
                        torch.Tensor([], device = self.device)
        )
        self._add_state("dist_2_samples",
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

        _wasserstein_param_check(new_samples_dist_1, new_weights_dist_1, 
                                 new_samples_dist_2, new_weights_dist_2
        )

        _wasserstein_update_input_check(new_samples_dist_1, new_weights_dist_1, 
                                        new_samples_dist_2, new_weights_dist_2
        )

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
        dist_1_samples = self.dist_1_samples
        dist_2_samples = self.dist_1_samples
        dist_1_weights = self.dist_1_weights
        dist_2_weights = self.dist_2_weights

        for metric in metrics:
            dist_1_samples = torch.cat((dist_1_samples, metric.dist_1_samples))
            dist_2_samples = torch.cat((dist_2_samples, metric.dist_2_samples))
            dist_1_weights = torch.cat((dist_1_weights, metric.dist_1_weights))
            dist_2_weights = torch.cat((dist_2_weights, metric.dist_2_weights))

        self.dist_1_samples = dist_1_samples
        self.dist_2_samples = dist_2_samples
        self.dist_1_weights = dist_1_weights
        self.dist_2_weights = dist_2_weights

        return self