from typing import Tuple, Optional, Union

import torch

@torch.inference_mode()
def wasserstein_1d(x: torch.Tensor, y: torch.Tensor,
                   x_weights: Optional[torch.Tensor]=None, y_weights: Optional[torch.Tensor]=None
) -> torch.Tensor:
    pass

def _validate_distribution(values: torch.Tensor, weights: torch.Tensor
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

    if values.nelement() == 0:
        raise ValueError("Distribution cannot be empty.")

    if weights is None:
        return values, None
    else:
        if weights.nelement() != values.nelement():
            raise ValueError("Value and weight tensor must be of the same size.")
        if torch.all(weights < 0):
            raise ValueError("All weights must be non-negative.")
        if not ( 0 < torch.sum(weights) < torch.inf ):
            raise ValueError("Weight tensor sum must be positive-finite.")
        
        return values, weights

def _cdf_distribution(p: int, x: torch.Tensor, y: torch.Tensor,
                   x_weights: Optional[torch.Tensor]=None, y_weights: Optional[torch.Tensor]=None
) -> torch.Tensor:
    pass