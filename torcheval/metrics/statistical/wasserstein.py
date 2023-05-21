import logging
from typing import Iterable, Optional, TypeVar, Union

import torch

from torcheval.metrics.functional.statistical.wasserstein import _validate_distribution, _cdf_distribution
from torcheval.metrics.metric import Metric

TWasserstein = TypeVar("TWasserstein")

class Wasserstein1D(Metric[torch.Tensor]):

    def __init__(
        self: TWasserstein,
        *,
        device: Optional[torch.device] = None
    ) -> None:
        super().__init__(device=device)
    
    @torch.inference_mode()
    def merge_state(self: TWasserstein, metrics: Iterable[TWasserstein]) -> TWasserstein:
        return super().merge_state(metrics)