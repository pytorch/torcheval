from typing import Tuple, Optional, Union

import torch

@torch.inference_mode()
def wasserstein_1d(x: torch.Tensor, y: torch.Tensor,
                   x_weights: Optional[torch.Tensor]=None, y_weights: Optional[torch.Tensor]=None
) -> torch.Tensor:
    pass

def _validate_distribution(values: torch.Tensor, weights: torch.Tensor
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    #values = torch.Tensor(values, dtype=torch.float)

    if values.nelement() == 0:
        raise ValueError("Distribution cannot be empty.")

    if weights is None:
        return values, None
    else:
        #weights = torch.Tensor(weights, dtype=torch.float)
        pass

def _cdf_distribution(p: int, x: torch.Tensor, y: torch.Tensor,
                   x_weights: Optional[torch.Tensor]=None, y_weights: Optional[torch.Tensor]=None
) -> torch.Tensor:
    pass