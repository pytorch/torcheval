from typing import Tuple, Optional, Union

import torch

@torch.inference_mode()
def wasserstein_1d(x: torch.Tensor, y: torch.Tensor,
                   x_weights: Optional[torch.Tensor]=None, y_weights: Optional[torch.Tensor]=None
) -> torch.Tensor:
    """
    Compute the first Wasserstein distance between two 1D distributions.

    This distance is also known as the earth mover's distance, since it can be
    seen as the minimum amount of "work" required to transform :math:`u` into
    :math:`v`, where "work" is measured as the amount of distribution weight
    that must be moved, multiplied by the distance it has to be moved.

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
    distance : float
        The computed distance between the distributions.

    Notes
    -----
    The first Wasserstein distance between the distributions :math:`u` and
    :math:`v` is:

    .. math::

        l_1 (u, v) = \inf_{\pi \in \Gamma (u, v)} \int_{\mathbb{R} \times
        \mathbb{R}} |x-y| \mathrm{d} \pi (x, y)

    where :math:`\Gamma (u, v)` is the set of (probability) distributions on
    :math:`\mathbb{R} \times \mathbb{R}` whose marginals are :math:`u` and
    :math:`v` on the first and second factors respectively.

    If :math:`U` and :math:`V` are the respective CDFs of :math:`u` and
    :math:`v`, this distance also equals to:

    .. math::

        l_1(u, v) = \int_{-\infty}^{+\infty} |U-V|

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
    >>> from scipy.stats import wasserstein_distance
    >>> wasserstein_distance([0, 1, 3], [5, 6, 8])
    5.0
    >>> wasserstein_distance([0, 1], [0, 1], [3, 1], [2, 2])
    0.25
    >>> wasserstein_distance([3.4, 3.9, 7.5, 7.8], [4.5, 1.4],
    ...                      [1.4, 0.9, 3.1, 7.2], [3.2, 3.5])
    4.0781331438047861

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
    
    # Compute the value of integral based on p = 1
    return torch.sum(torch.multiply(torch.abs(x_cdf - y_cdf), deltas))
