from typing import Tuple, Optional, Union

import torch

@torch.inference_mode()
def wasserstein_1d(x: torch.Tensor, y: torch.Tensor,
                   x_weights: Optional[torch.Tensor]=None, y_weights: Optional[torch.Tensor]=None
) -> torch.Tensor:
    # Ensuring values are from a valid distribution
    x, x_weights = _validate_distribution(x, x_weights)
    y, y_weights = _validate_distribution(y, y_weights)

    # Finding the sorted values
    x_sorter = torch.argsort(x)
    y_sorter = torch.argsort(y)

    # Bringing all the values on a central number line
    all_values = torch.concatenate((x, y))
    all_values = torch.sort(all_values)

    # Compute the differences between successive values of x and y
    deltas = torch.diff(all_values)

    # Obtain respective positions of the x and y values among all_values
    x_cdf_indices = torch.searchsorted(x[x_sorter], all_values)
    y_cdf_indices = torch.searchsorted(y[y_sorter], all_values)

    # Calculate the CDF of x and y using their weights, if specified
    if x_weights is None:
        x_cdf = x_cdf_indices / x.size()[0]
    else:
        x_sorted_cum_weights = torch.cat(([0],
                                         torch.cumsum(x[x_sorter])))
        x_cdf = x_sorted_cum_weights[x_cdf_indices] / x_sorted_cum_weights[-1]
    
    if y_weights is None:
        y_cdf = y_cdf_indices / y.size()[0]
    else:
        y_sorted_cum_weights = torch.cat(([0],
                                         torch.cumsum(y[y_sorter])))
        y_cdf = y_sorted_cum_weights[y_cdf_indices] / y_sorted_cum_weights[-1]
    
    # Compute the value of integral based on p = 1
    return torch.sum(torch.multiply(torch.abs(x_cdf - y_cdf), deltas))

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
