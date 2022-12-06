# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]: Pyre was not able to infer the type of argument

from typing import Tuple

import numpy as np

import torch

from sklearn.metrics import average_precision_score as sk_ap

from torcheval.metrics import BinaryAUPRC, MulticlassAUPRC
from torcheval.utils.test_utils.metric_class_tester import (
    BATCH_SIZE,
    MetricClassTester,
    NUM_PROCESSES,
)

NUM_TOTAL_UPDATES = 8


class TestBinaryAUPRC(MetricClassTester):
    def _get_sklearn_equivalent(
        self, input: torch.Tensor, target: torch.Tensor, device: str = "cpu"
    ) -> torch.Tensor:
        # Convert input/target to sklearn style inputs
        # run each task once at a time since no multi-task/multiclass
        # available for sklearn
        skinputs = torch.permute(input, (1, 0, 2)).flatten(1).numpy()
        sktargets = torch.permute(target, (1, 0, 2)).flatten(1).numpy()
        auprcs = []
        for i in range(skinputs.shape[0]):
            skinput = skinputs[i, :]
            sktarget = sktargets[i, :]
            auprcs.append(np.nan_to_num(sk_ap(sktarget, skinput)))
        return torch.tensor(auprcs, device=device).to(torch.float32)

    def _get_rand_inputs_binary(
        self, num_updates: int, num_tasks: int, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input = torch.rand(size=[num_updates, num_tasks, batch_size])
        targets = torch.randint(
            low=0, high=2, size=[num_updates, num_tasks, batch_size]
        )
        return input, targets

    def _check_against_sklearn(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_tasks: int,
        num_updates: int = NUM_TOTAL_UPDATES,
    ) -> None:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        compute_result = self._get_sklearn_equivalent(input, target, device)
        self.run_class_implementation_tests(
            metric=BinaryAUPRC(num_tasks=num_tasks),
            state_names={"inputs", "targets"},
            update_kwargs={
                "input": input,
                "target": target,
            },
            compute_result=compute_result,
            num_processes=NUM_PROCESSES,
            num_total_updates=NUM_TOTAL_UPDATES,
        )

    def test_binary_auroc_class_with_good_inputs(self) -> None:
        num_tasks = 2
        input, target = self._get_rand_inputs_binary(
            NUM_TOTAL_UPDATES, num_tasks, BATCH_SIZE
        )
        self._check_against_sklearn(input, target, num_tasks)

        num_tasks = 4
        input, target = self._get_rand_inputs_binary(
            NUM_TOTAL_UPDATES, num_tasks, BATCH_SIZE
        )
        self._check_against_sklearn(input, target, num_tasks)

        num_tasks = 8
        input, target = self._get_rand_inputs_binary(
            NUM_TOTAL_UPDATES, num_tasks, BATCH_SIZE
        )
        self._check_against_sklearn(input, target, num_tasks)

    def test_binary_auprc_docstring_examples(self) -> None:
        metric = BinaryAUPRC()
        input = torch.tensor([0.1, 0.5, 0.7, 0.8])
        target = torch.tensor([1, 0, 1, 1])
        metric.update(input, target)
        result = metric.compute()
        expected = torch.tensor(0.9167)
        torch.testing.assert_close(
            result,
            expected,
            equal_nan=True,
            atol=1e-4,
            rtol=1e-3,
        )

        metric = BinaryAUPRC()
        input = torch.tensor([[0.5, 2]])
        target = torch.tensor([[0, 0]])
        metric.update(input, target)
        result = metric.compute()
        expected = torch.tensor([-0.0])
        torch.testing.assert_close(
            result,
            expected,
            equal_nan=True,
            atol=1e-4,
            rtol=1e-3,
        )

        input = torch.tensor([[2, 1.5]])
        target = torch.tensor([[1, 0]])
        metric.update(input, target)
        result = metric.compute()
        expected = torch.tensor([0.5000])
        torch.testing.assert_close(
            result,
            expected,
            equal_nan=True,
            atol=1e-4,
            rtol=1e-3,
        )

        metric = BinaryAUPRC(num_tasks=3)
        input = torch.tensor([[0.1, 0, 0.1, 0], [0, 1, 0.2, 0], [0, 0, 0.7, 1]])
        target = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]])
        metric.update(input, target)
        result = metric.compute()
        expected = torch.tensor([0.5000, 1.0000, 1.0000])
        torch.testing.assert_close(
            result,
            expected,
            equal_nan=True,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_binary_auroc_class_invalid_input(self) -> None:
        metric = BinaryAUPRC()
        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same shape, "
            r"got shapes torch.Size\(\[4\]\) and torch.Size\(\[3\]\).",
        ):
            metric.update(torch.rand(4), torch.rand(3))

        metric = BinaryAUPRC()
        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks = 1`, `input` and `target` are expected to be one-dimensional tensors or 1xN tensors, but got shape input: "
            r"torch.Size\(\[4, 5\]\), target: torch.Size\(\[4, 5\]\).",
        ):
            metric.update(torch.rand(4, 5), torch.rand(4, 5))

        metric = BinaryAUPRC()
        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks = 1`, `input` and `target` are expected to be one-dimensional tensors or 1xN tensors, but got shape input: "
            r"torch.Size\(\[4, 5, 5\]\), target: torch.Size\(\[4, 5, 5\]\).",
        ):
            metric.update(torch.rand(4, 5, 5), torch.rand(4, 5, 5))

        metric = BinaryAUPRC(num_tasks=2)
        with self.assertRaisesRegex(
            ValueError,
            r"`num_tasks = 2`, `input` and `target` shape is expected to be \(2, num_samples\), but got shape input: "
            r"torch.Size\(\[4, 5\]\), target: torch.Size\(\[4, 5\]\).",
        ):
            metric.update(torch.rand(4, 5), torch.rand(4, 5))

        with self.assertRaisesRegex(
            ValueError, "`num_tasks` must be an integer greater than or equal to 1"
        ):
            metric = BinaryAUPRC(num_tasks=0)


class TestMulticlassAUPRC(MetricClassTester):
    def _get_sklearn_equivalent(
        self, input: torch.Tensor, target: torch.Tensor, device: str = "cpu"
    ) -> torch.Tensor:
        # concat updates to into batch dimension
        skinputs = input.reshape(-1, input.shape[2]).numpy()
        sktargets = target.reshape(-1).numpy()

        # Convert input/target to sklearn style inputs
        # run each task once at a time since no multi-task/multiclass
        # available for sklearn
        auprcs = []
        for i in range(skinputs.shape[1]):
            skinput = skinputs[:, i]
            sktarget = np.where(sktargets == i, 1, 0)
            auprcs.append(np.nan_to_num(sk_ap(sktarget, skinput)))
        return torch.tensor(auprcs, device=device).to(torch.float32)

    def _get_rand_inputs_multiclass(
        self, num_updates: int, num_classes: int, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input = torch.rand(size=[num_updates, batch_size, num_classes])
        targets = torch.randint(low=0, high=num_classes, size=[num_updates, batch_size])
        return input, targets

    def _check_against_sklearn(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        num_updates: int = NUM_TOTAL_UPDATES,
    ) -> None:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        compute_result = self._get_sklearn_equivalent(input, target, device)
        self.run_class_implementation_tests(
            metric=MulticlassAUPRC(num_classes=num_classes, average=None),
            state_names={"inputs", "targets"},
            update_kwargs={
                "input": input,
                "target": target,
            },
            compute_result=compute_result,
            num_processes=NUM_PROCESSES,
            num_total_updates=NUM_TOTAL_UPDATES,
        )

    def test_multiclass_auroc_class_with_good_inputs(self) -> None:
        num_classes = 2
        input, target = self._get_rand_inputs_multiclass(
            NUM_TOTAL_UPDATES, num_classes, BATCH_SIZE
        )
        self._check_against_sklearn(input, target, num_classes)

        num_classes = 4
        input, target = self._get_rand_inputs_multiclass(
            NUM_TOTAL_UPDATES, num_classes, BATCH_SIZE
        )
        self._check_against_sklearn(input, target, num_classes)

        num_classes = 8
        input, target = self._get_rand_inputs_multiclass(
            NUM_TOTAL_UPDATES, num_classes, BATCH_SIZE
        )
        self._check_against_sklearn(input, target, num_classes)

    def test_multiclass_auprc_docstring_examples(self) -> None:
        metric = MulticlassAUPRC(num_classes=3)
        input = torch.tensor(
            [[0.1, 0.1, 0.1], [0.5, 0.5, 0.5], [0.7, 0.7, 0.7], [0.8, 0.8, 0.8]]
        )
        target = torch.tensor([0, 2, 1, 1])
        metric.update(input, target)
        result = metric.compute()
        expected = torch.tensor(0.5278)
        torch.testing.assert_close(
            result,
            expected,
            equal_nan=True,
            atol=1e-4,
            rtol=1e-3,
        )

        metric = MulticlassAUPRC(num_classes=3)
        input = torch.tensor([[0.5, 0.2, 3], [2, 1, 6]])
        target = torch.tensor([0, 2])
        metric.update(input, target)
        result = metric.compute()
        expected = torch.tensor(0.5000)
        torch.testing.assert_close(
            result,
            expected,
            equal_nan=True,
            atol=1e-4,
            rtol=1e-3,
        )

        input = torch.tensor([[5, 3, 2], [0.2, 2, 3], [3, 3, 3]])
        target = torch.tensor([2, 2, 1])
        metric.update(input, target)
        result = metric.compute()
        expected = torch.tensor(0.4833)
        torch.testing.assert_close(
            result,
            expected,
            equal_nan=True,
            atol=1e-4,
            rtol=1e-3,
        )

        metric = MulticlassAUPRC(num_classes=3, average=None)
        input = torch.tensor([[0.1, 0, 0], [0, 1, 0], [0.1, 0.2, 0.7], [0, 0, 1]])
        target = torch.tensor([0, 1, 2, 2])
        metric.update(input, target)
        result = metric.compute()
        expected = torch.tensor([0.5000, 1.0000, 1.0000])
        torch.testing.assert_close(
            result,
            expected,
            equal_nan=True,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_multiclass_auprc_options(self) -> None:
        # average = macro (implicit)
        num_classes = 4
        input, target = self._get_rand_inputs_multiclass(
            NUM_TOTAL_UPDATES, num_classes, BATCH_SIZE
        )
        self._check_against_sklearn(input, target, num_classes)

        compute_result = torch.mean(self._get_sklearn_equivalent(input, target, "cpu"))

        self.run_class_implementation_tests(
            metric=MulticlassAUPRC(num_classes=num_classes),
            state_names={"inputs", "targets"},
            update_kwargs={
                "input": input,
                "target": target,
            },
            compute_result=compute_result,
            num_processes=NUM_PROCESSES,
            num_total_updates=NUM_TOTAL_UPDATES,
        )

        # average = macro (explicit)
        num_classes = 4
        input, target = self._get_rand_inputs_multiclass(
            NUM_TOTAL_UPDATES, num_classes, BATCH_SIZE
        )
        self._check_against_sklearn(input, target, num_classes)

        compute_result = torch.mean(self._get_sklearn_equivalent(input, target, "cpu"))

        self.run_class_implementation_tests(
            metric=MulticlassAUPRC(num_classes=num_classes, average="macro"),
            state_names={"inputs", "targets"},
            update_kwargs={
                "input": input,
                "target": target,
            },
            compute_result=compute_result,
            num_processes=NUM_PROCESSES,
            num_total_updates=NUM_TOTAL_UPDATES,
        )

        # average = None
        num_classes = 4
        input, target = self._get_rand_inputs_multiclass(
            NUM_TOTAL_UPDATES, num_classes, BATCH_SIZE
        )
        self._check_against_sklearn(input, target, num_classes)

        compute_result = self._get_sklearn_equivalent(input, target, "cpu")

        self.run_class_implementation_tests(
            metric=MulticlassAUPRC(num_classes=num_classes, average=None),
            state_names={"inputs", "targets"},
            update_kwargs={
                "input": input,
                "target": target,
            },
            compute_result=compute_result,
            num_processes=NUM_PROCESSES,
            num_total_updates=NUM_TOTAL_UPDATES,
        )
        self.run_class_implementation_tests(
            metric=MulticlassAUPRC(num_classes=num_classes, average="none"),
            state_names={"inputs", "targets"},
            update_kwargs={
                "input": input,
                "target": target,
            },
            compute_result=compute_result,
            num_processes=NUM_PROCESSES,
            num_total_updates=NUM_TOTAL_UPDATES,
        )

    def test_multiclass_auroc_class_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "`average` was not in the allowed value of .*, got micro."
        ):
            MulticlassAUPRC(num_classes=4, average="micro")

        with self.assertRaisesRegex(ValueError, "`num_classes` has to be at least 2."):
            MulticlassAUPRC(num_classes=1)

        metric = MulticlassAUPRC(num_classes=2)
        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same first dimension, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            metric.update(torch.rand(4, 2), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "target should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            metric.update(torch.rand(3, 2), torch.rand(3, 2))

        with self.assertRaisesRegex(
            ValueError,
            r"input should have shape of \(num_sample, num_classes\), "
            r"got torch.Size\(\[3, 4\]\) and num_classes=2.",
        ):
            metric.update(torch.rand(3, 4), torch.rand(3))
