# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch


@torch.inference_mode()
def retrieval_precision(
    input: torch.Tensor,
    target: torch.Tensor,
    k: Optional[int] = None,
    limit_k_to_size: bool = False,
) -> torch.Tensor:
    """
    Retrieval Precision is a metric that measures the proportion of relevant items retrieved out of the total items retrieved by an information retrieval system.

    It is defined as:
    Retrieval Precision = (Number of Relevant Items Retrieved) / (Total Number of Items Retrieved)
    This metric is also known as Precision at k, where k is the number of elements considered as being retrieved.

    Its class version is :class:`torcheval.metrics.ranking.RetrievalPrecision`.

    Args:
        input (Tensor):
            Predicted scores for each document (the higher the more relevant), with shape (num_sample,) or (num_tasks, num_samples).
        target (Tensor):
            0 and 1 valued Tensor of ground truth identifying relevant element, with shape (num_sample,) or (num_tasks, num_samples).
        k (int, optional):
            the number of elements considered as being retrieved. Only the top (sorted in decreasing order) `k` elements of `input` are considered.
            if `k` is None, all the `input` elements are considered.
        limit_k_to_size (bool, default value: False):
            When set to `True`, limits `k` to be at most the length of `input`, i.e. replaces `k` by `k=min(k, len(input))`.
            This parameter can only be set to `True` if `k` is not None.
    Returns:
       (Tensor):
            - If input and target are 1D: returns a tensor of dimension 0, containing the retrieval precision value.
            - When input and target are 2D with shape (num_tasks, num_samples): returns a tensor of shape (num_tasks,) containing the retrieval precision, computed row by row.

    Examples (one dimension):
        >>> import torch
        >>> from torcheval.metrics.functional.ranking import retrieval_precision

        >>> input = tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
        >>> target = tensor([0, 0, 1, 1, 1, 0, 1])
        >>> retrieval_precision(input, target)
        tensor(0.571)
        >>> retrieval_precision(input, target, k=2)
        tensor(0.5)
        >>> retrieval_precision(input, target, k=4)
        tensor(0.5)
        >>> retrieval_precision(input, target, k=10)
        tensor(0.400)
        >>> retrieval_precision(input, target, k=10, limit_k_to_size=True)
        tensor(0.571)

    Examples (two dimensions):
        >>> import torch
        >>> from torcheval.metrics.functional.ranking import retrieval_precision

        >>> input = tensor([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
        >>> target = tensor([[0, 0, 1], [1, 0, 0]])
        >>> retrieval_precision(input, target, k=2)
        tensor([0.5000, 0.0000])

    Raises:
        ValueError:
            if `limit_k_to_size` is True and `k` is None.
        ValueError:
            if `k` is not a positive integer.
        ValueError:
            if input or target arguments of self._retrieval_precision_compute are Tensors with dimension 0 or > 2.
    """
    _retrieval_precision_param_check(k, limit_k_to_size)
    _retrieval_precision_input_check(input, target)
    return _retrieval_precision_compute(
        input=input,
        target=target,
        k=k,
        limit_k_to_size=limit_k_to_size,
    )


def _retrieval_precision_param_check(
    k: Optional[int] = None, limit_k_to_size: bool = False
) -> None:
    if k is not None and k <= 0:
        raise ValueError(f"k must be a positive integer, got k={k}.")

    if limit_k_to_size and k is None:
        raise ValueError(
            "when limit_k_to_size is True, k must be a positive (>0) integer."
        )


def _retrieval_precision_input_check(input: torch.Tensor, target: torch.Tensor) -> None:
    for obj, obj_name in [(input, "input"), (target, "target")]:
        if not (1 <= obj.dim() <= 2):
            raise ValueError(
                f"{obj_name} should be a one or two dimensional tensor, got {obj_name}.dim()={obj.dim()}."
            )

    if input.shape != target.shape:
        raise ValueError(
            f"input and target must be of the same shape, got input.shape={input.shape} and target.shape={target.shape}."
        )


def _retrieval_precision_compute(
    input: torch.Tensor,
    target: torch.Tensor,
    k: Optional[int] = None,
    limit_k_to_size: bool = False,
) -> torch.Tensor:
    nb_samples = input.size(-1)

    if k is None:
        nb_retrieved_items = k = nb_samples

    elif limit_k_to_size:
        nb_retrieved_items = min(k, nb_samples)

    else:
        nb_retrieved_items = k

    # take the topk values of input. /!\ Ties are sorted in an unpredictable way.
    topk_idx = input.topk(min(k, nb_samples), dim=-1)[1]

    return target.gather(dim=-1, index=topk_idx).sum(dim=-1) / nb_retrieved_items
