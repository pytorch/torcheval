from typing import Tuple, Optional, Union

import torch

@torch.inference_mode()
def wasserstein_1d(x: torch.Tensor, y: torch.Tensor,
                   x_weights: Optional[torch.Tensor]=None,
                   y_weights: Optional[torch.Tensor]=None
) -> torch.Tensor:
    """
    The Wasserstein distance, also called the Earth Mover's Distance, is a 
    measure of the similarity between two distributions.

    The Wasserstein distance between two distributions is intuitively the 
    minimum weight of soil (times distance moved) that would need to be moved 
    if the two distributions were represented by two piles of soil.

    Args
    ----------
    x, y (Tensor) : Tensor values observed in the distribution.
    x_weights, y_weights (Tensor): Optional tensor weights for each value.
        If unspecified, each value is assigned the same value.
        `x_weights` (resp. `y_weights`) must have the same length as
        `x` (resp. `y`). If the weight sum differs from 1, it
        must still be positive and finite so that the weights can be normalized
        to sum to 1.

    Returns
    -------
    distance : Tensor value
        The computed distance between the distributions.

    Notes
    -----
    The first Wasserstein distance between the distributions :math:`x` and
    :math:`x` is:

    .. math::

        W_1 (x, y) = \inf_{\pi \in \Gamma (x, y)} \int_{\mathbb{R} \times
        \mathbb{R}} |p-q| \mathrm{d} \pi (p, q)

    where :math:`\Gamma (x, y)` is the set of (probability) distributions on
    :math:`\mathbb{R} \times \mathbb{R}` whose marginals are :math:`x` and
    :math:`y` on the first and second factors respectively.

    If :math:`X` and :math:`Y` are the respective CDFs of :math:`x` and
    :math:`y`, this distance also equals to:

    .. math::

        W_1(x, y) = \int_{-\infty}^{+\infty} |X-Y|

    See [2]_ for a proof of the equivalence of both definitions.

    The input distributions can be empirical, therefore coming from samples
    whose values are effectively inputs of the function, or they can be seen as
    generalized functions, in which case they are weighted sums of Dirac delta
    functions located at the specified values.

    References
    ----------
    .. [1] "Wasserstein metric", https://en.wikipedia.org/wiki/Wasserstein_metric
    .. [2] Ramdas, Garcia, Cuturi "On Wasserstein Two Sample Testing and Related
           Families of Nonparametric Tests" (2015). :arXiv:`1509.02237`.

    Examples
    --------
    >>> from torcheval.metrics.functional import wasserstein_1d
    >>> wasserstein_1d(torch.tensor([0,1,2]), torch.tensor([0,1,1]))
    torch.tensor([0.33333333333333337])
    >>> wasserstein_1d(torch.tensor([0,1,2]), torch.tensor([0,1,1]), torch.tensor([1,2,0]), torch.tensor([1,1,1]))
    torch.tensor([0.0])
    >>> wasserstein_1d(torch.tensor([0,1,2,2]), torch.tensor([0,1]))
    torch.tensor([0.75])

    """
    _wasserstein_param_check(x, x_weights)
    _wasserstein_param_check(y, y_weights)
    _wasserstein_update_input_check(x, x_weights)
    _wasserstein_update_input_check(y, y_weights)
    return _wasserstein_compute(x, x_weights, y, y_weights)

@torch.inference_mode()
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
    
def _wasserstein_param_check(
        values: torch.Tensor,
        weights: torch.Tensor
) -> None:
    if values.nelement() == 0:
        raise ValueError("Distribution cannot be empty.")
    if weights is not None and weights.nelement() == 0:
        raise ValueError("Weights cannot be empty.")

def _wasserstein_update_input_check(
        values: torch.Tensor,
        weights: torch.Tensor
) -> None:
    if weights is not None:
        if weights.nelement() != values.nelement():
            raise ValueError("Value and weight tensor must be of the same size.")
        if torch.all(weights < 0):
            raise ValueError("All weights must be non-negative.")
        if not ( 0 < torch.sum(weights) < torch.inf ):
            raise ValueError("Weight tensor sum must be positive-finite.")

@torch.jit.script
def _wasserstein_compute(
        x: torch.Tensor,
        y: torch.Tensor,
        x_weights: Optional[torch.Tensor],
        y_weights: Optional[torch.Tensor]
) -> torch.Tensor:
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
    

    return torch.sum(torch.multiply(torch.abs(x_cdf - y_cdf), deltas))
