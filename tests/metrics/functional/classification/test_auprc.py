# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
import unittest
from typing import Optional, Tuple

import numpy as np

import torch
from sklearn.metrics import average_precision_score as sk_ap
from torcheval.metrics.functional import binary_auprc, multiclass_auprc
from torcheval.utils.test_utils.metric_class_tester import BATCH_SIZE


class TestBinaryAUPRC(unittest.TestCase):
    def _get_sklearn_equivalent(
        self, input: torch.Tensor, target: torch.Tensor, device: str = "cpu"
    ) -> torch.Tensor:

        # Convert input/target to sklearn style inputs
        # run each task once at a time since no multi-task/multiclass
        # available for sklearn
        skinputs = input.numpy()
        sktargets = target.numpy()
        sk_auprcs = []
        for i in range(skinputs.shape[0]):
            skinput = skinputs[i, :]
            sktarget = sktargets[i, :]
            sk_auprcs.append(np.nan_to_num(sk_ap(sktarget, skinput)))
        return torch.tensor(sk_auprcs, device=device).to(torch.float32)

    def _test_binary_auprc_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_tasks: int = 1,
        compute_result: Optional[torch.Tensor] = None,
    ) -> None:

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        if compute_result is None:
            compute_result = self._get_sklearn_equivalent(input, target, device)

        # Get torcheval compute result
        te_compute_result = binary_auprc(
            input.to(device=device),
            target.to(device=device),
            num_tasks=num_tasks,
        )

        torch.testing.assert_close(
            te_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def _get_rand_inputs_binary(
        self, num_tasks: int, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input = torch.rand(size=[num_tasks, batch_size])
        targets = torch.randint(low=0, high=2, size=[num_tasks, batch_size])
        return input, targets

    def test_binary_auprc_docstring_examples(self) -> None:
        """This test also tests whether the shape is preserved for 1D and 2D examples with ntasks=1"""
        input = torch.tensor([0.1, 0.5, 0.7, 0.8])
        target = torch.tensor([1, 0, 1, 1])
        compute_result = torch.tensor(0.9167)

        torch.testing.assert_close(
            binary_auprc(input, target),
            compute_result,
            atol=1e-4,
            rtol=1e-4,
        )

        input = torch.tensor([[1, 1, 1, 0]])
        target = torch.tensor([[1, 0, 1, 0]])
        compute_result = torch.tensor([0.6667])

        torch.testing.assert_close(
            binary_auprc(input, target),
            compute_result,
            atol=1e-4,
            rtol=1e-4,
        )

        input = torch.tensor([[0.1, 0.5, 0.7, 0.8], [1, 1, 1, 0]])
        target = torch.tensor([[1, 0, 1, 1], [1, 0, 1, 0]])
        compute_result = torch.tensor([0.9167, 0.6667])

        torch.testing.assert_close(
            binary_auprc(input, target, num_tasks=2),
            compute_result,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_binary_auprc_with_good_input(self) -> None:
        # one task
        input, target = self._get_rand_inputs_binary(1, BATCH_SIZE)
        self._test_binary_auprc_with_input(input, target, num_tasks=1)

        # two tasks
        input, target = self._get_rand_inputs_binary(2, BATCH_SIZE)
        self._test_binary_auprc_with_input(input, target, num_tasks=2)

        # random number of tasks between 3 and 10
        num_tasks = random.randint(3, 10)
        input, target = self._get_rand_inputs_binary(num_tasks, BATCH_SIZE)
        self._test_binary_auprc_with_input(input, target, num_tasks=num_tasks)

    def test_binary_auprc_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same shape, "
            r"got shapes torch.Size\(\[4\]\) and torch.Size\(\[3\]\).",
        ):
            binary_auprc(torch.rand(4), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks = 1`, `input` and `target` are expected to be one-dimensional tensors or 1xN tensors, but got shape input: "
            r"torch.Size\(\[4, 5\]\), target: torch.Size\(\[4, 5\]\).",
        ):
            binary_auprc(torch.rand(4, 5), torch.rand(4, 5))

        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks = 1`, `input` and `target` are expected to be one-dimensional tensors or 1xN tensors, but got shape input: "
            r"torch.Size\(\[4, 5, 5\]\), target: torch.Size\(\[4, 5, 5\]\).",
        ):
            binary_auprc(torch.rand(4, 5, 5), torch.rand(4, 5, 5))

        with self.assertRaisesRegex(
            ValueError,
            r"`num_tasks = 2`, `input` and `target` shape is expected to be \(2, num_samples\), but got shape input: "
            r"torch.Size\(\[4, 5\]\), target: torch.Size\(\[4, 5\]\).",
        ):
            binary_auprc(torch.rand(4, 5), torch.rand(4, 5), num_tasks=2)


class TestMulticlassAUPRC(unittest.TestCase):
    def _get_sklearn_equivalent(
        self, input: torch.Tensor, target: torch.Tensor, device: str = "cpu"
    ) -> torch.Tensor:
        # Convert input/target to sklearn style inputs
        # run each task once at a time since no multi-task/multiclass
        # available for sklearn
        skinputs = input.numpy()
        sktargets = target.numpy()
        auprcs = []
        for i in range(input.shape[1]):
            skinput = skinputs[:, i]
            sktarget = np.where(sktargets == i, 1, 0)
            auprcs.append(np.nan_to_num(sk_ap(sktarget, skinput)))
        return torch.tensor(auprcs, device=device).to(torch.float32)

    def _test_multiclass_auprc_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        compute_result: Optional[torch.Tensor] = None,
    ) -> None:

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        # get sklearn compute result if none given
        if compute_result is None:
            compute_result = self._get_sklearn_equivalent(input, target, device)

        # Get torcheval compute result
        te_compute_result = multiclass_auprc(
            input.to(device=device),
            target.to(device=device),
            num_classes=num_classes,
            average=None,
        )

        # test no average
        torch.testing.assert_close(
            te_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def _get_rand_inputs_multiclass(
        self, num_classes: int, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input = torch.rand(size=[batch_size, num_classes])
        targets = torch.randint(low=0, high=num_classes, size=[batch_size])
        return input, targets

    def test_multiclass_auprc_with_good_input(self) -> None:
        num_classes = 2
        input, target = self._get_rand_inputs_multiclass(num_classes, BATCH_SIZE)
        self._test_multiclass_auprc_with_input(input, target, num_classes)

        num_classes = 4
        input, target = self._get_rand_inputs_multiclass(num_classes, BATCH_SIZE)
        self._test_multiclass_auprc_with_input(input, target, num_classes)

        num_classes = 5
        input, target = self._get_rand_inputs_multiclass(num_classes, BATCH_SIZE)
        self._test_multiclass_auprc_with_input(input, target, num_classes)

        num_classes = 8
        input, target = self._get_rand_inputs_multiclass(num_classes, BATCH_SIZE)
        self._test_multiclass_auprc_with_input(input, target, num_classes)

    def test_multiclass_auprc_options(self) -> None:
        # average = macro, num_classes not given
        input, target = self._get_rand_inputs_multiclass(5, BATCH_SIZE)
        compute_result = torch.mean(self._get_sklearn_equivalent(input, target))
        te_compute_result = multiclass_auprc(input, target, average="macro")
        torch.testing.assert_close(
            te_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

        # average = macro (not given), num_classes given
        input, target = self._get_rand_inputs_multiclass(5, BATCH_SIZE)
        compute_result = torch.mean(self._get_sklearn_equivalent(input, target))
        te_compute_result = multiclass_auprc(input, target)
        print(compute_result, te_compute_result)
        torch.testing.assert_close(
            te_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

        # average = none
        input, target = self._get_rand_inputs_multiclass(5, BATCH_SIZE)
        compute_result = self._get_sklearn_equivalent(input, target)
        te_compute_result = multiclass_auprc(input, target, average="none")
        torch.testing.assert_close(
            te_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )
        te_compute_result = multiclass_auprc(input, target, average=None)
        torch.testing.assert_close(
            te_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_multiclass_auprc_docstring_examples(self) -> None:
        input = torch.tensor([[0.5647, 0.2726], [0.9143, 0.1895], [0.7782, 0.3082]])
        target = torch.tensor([0, 1, 0])
        output = torch.tensor([0.5833, 0.3333])
        result = multiclass_auprc(input, target, average=None)
        torch.testing.assert_close(
            result,
            output,
            equal_nan=True,
            atol=1e-4,
            rtol=1e-3,
        )

        avg_result = multiclass_auprc(input, target)
        avg_output = torch.tensor(0.4583)
        torch.testing.assert_close(
            avg_result,
            avg_output,
            equal_nan=True,
            atol=1e-4,
            rtol=1e-3,
        )

        input = torch.tensor([[0.1, 1], [0.5, 1], [0.7, 1], [0.8, 0]])
        target = torch.tensor([1, 0, 0, 1])
        result = multiclass_auprc(input, target, 2, average=None)
        output = torch.tensor([0.5833, 0.4167])
        torch.testing.assert_close(
            avg_result,
            avg_output,
            equal_nan=True,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_multiclass_auroc_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "`average` was not in the allowed value of .*, got micro."
        ):
            num_classes = 4
            input, target = self._get_rand_inputs_multiclass(num_classes, BATCH_SIZE)
            multiclass_auprc(
                input,
                target,
                num_classes=num_classes,
                average="micro",
            )

        with self.assertRaisesRegex(ValueError, "`num_classes` has to be at least 2."):
            multiclass_auprc(torch.rand(4, 2), torch.rand(2), num_classes=1)

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same first dimension, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            multiclass_auprc(torch.rand(4, 2), torch.rand(3), num_classes=2)

        with self.assertRaisesRegex(
            ValueError,
            "target should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            multiclass_auprc(torch.rand(3, 2), torch.rand(3, 2), num_classes=2)

        with self.assertRaisesRegex(
            ValueError,
            r"input should have shape of \(num_sample, num_classes\), "
            r"got torch.Size\(\[3, 4\]\) and num_classes=2.",
        ):
            multiclass_auprc(torch.rand(3, 4), torch.rand(3), num_classes=2)
