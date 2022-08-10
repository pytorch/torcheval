#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torchvision.models as models
from torcheval.tools.module_summary import _get_human_readable_count, get_module_summary


class ModuleSummaryTest(unittest.TestCase):
    def test_module_summary_layer(self) -> None:
        """Make sure ModuleSummary works for a single layer."""
        model = torch.nn.Conv2d(3, 8, 3)
        ms1 = get_module_summary(model)
        ms2 = get_module_summary(model, max_depth=1)
        ms3 = get_module_summary(
            model, max_depth=1, module_input=torch.randn(1, 3, 8, 8)
        )

        self.assertEqual(ms1.module_name, "")
        self.assertEqual(ms1.module_type, "Conv2d")
        self.assertEqual(ms1.num_parameters, 224)
        self.assertEqual(ms1.num_trainable_parameters, 224)
        self.assertEqual(ms1.size_bytes, 224 * 4)
        self.assertEqual(ms1.submodule_summaries, {})
        self.assertFalse(ms1.has_uninitialized_param)

        self.assertEqual(ms1.module_name, ms2.module_name)
        self.assertEqual(ms1.module_type, ms2.module_type)
        self.assertEqual(ms1.num_parameters, ms2.num_parameters)
        self.assertEqual(ms1.num_trainable_parameters, ms2.num_trainable_parameters)
        self.assertEqual(ms1.size_bytes, ms2.size_bytes)
        self.assertEqual(ms1.submodule_summaries, ms2.submodule_summaries)

        self.assertEqual(ms1.module_name, ms3.module_name)
        self.assertEqual(ms1.module_type, ms3.module_type)
        self.assertEqual(ms1.num_parameters, ms3.num_parameters)
        self.assertEqual(ms1.num_trainable_parameters, ms3.num_trainable_parameters)
        self.assertEqual(ms1.size_bytes, ms3.size_bytes)
        self.assertEqual(ms1.submodule_summaries, ms3.submodule_summaries)

        self.assertEqual(ms3.flops_forward, 7776)
        self.assertEqual(ms3.flops_backward, 7776)

    def test_flops_with_Batch(self) -> None:
        """Make sure FLOPs calculate are the same when input data has different batch size."""
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.Conv2d(8, 5, 3))
        ms1 = get_module_summary(
            model, max_depth=2, module_input=torch.randn(1, 3, 8, 8)
        )
        ms3 = get_module_summary(
            model, max_depth=2, module_input=torch.randn(3, 3, 8, 8)
        )
        self.assertEqual(ms3.flops_forward, 13536)
        self.assertEqual(ms3.flops_backward, 19296)
        self.assertEqual(ms1.flops_forward, 13536)
        self.assertEqual(ms1.flops_backward, 19296)
        self.assertEqual(ms3.submodule_summaries["0"].flops_forward, 7776)
        self.assertEqual(ms3.submodule_summaries["0"].flops_backward, 7776)
        self.assertEqual(ms1.submodule_summaries["0"].flops_forward, 7776)
        self.assertEqual(ms1.submodule_summaries["0"].flops_backward, 7776)
        self.assertEqual(ms3.submodule_summaries["1"].flops_forward, 5760)
        self.assertEqual(ms3.submodule_summaries["1"].flops_backward, 11520)
        self.assertEqual(ms1.submodule_summaries["1"].flops_forward, 5760)
        self.assertEqual(ms1.submodule_summaries["1"].flops_backward, 11520)

    def test_invalid_max_depth(self) -> None:
        """Test for ValueError when providing bad max_depth"""
        model = torch.nn.Conv2d(3, 8, 3)
        with self.assertRaisesRegex(ValueError, "Got -2."):
            get_module_summary(model, max_depth=-2)
        with self.assertRaisesRegex(ValueError, "Got 0."):
            get_module_summary(model, max_depth=0)

    def test_lazy_tensor(self) -> None:
        """Check for warning when passing in a lazy weight Tensor"""
        model = torch.nn.LazyLinear(10)
        ms = get_module_summary(model)
        with self.assertWarns(Warning):
            ms.num_parameters
        with self.assertWarns(Warning):
            ms.num_trainable_parameters
        self.assertTrue(ms.has_uninitialized_param)

    def test_lazy_tensor_flops(self) -> None:
        """Check for warnings when passing in a lazy weight Tensor
        Even when asking for flops calculation."""
        model = torch.nn.LazyLinear(10)
        ms = get_module_summary(model, module_input=torch.randn(1, 10))
        with self.assertWarns(Warning):
            ms.num_parameters
        with self.assertWarns(Warning):
            ms.num_trainable_parameters
        self.assertTrue(ms.has_uninitialized_param)
        self.assertEqual(ms.flops_backward, -1)
        self.assertEqual(ms.flops_forward, -1)

    def test_resnet_max_depth(self) -> None:
        """Test the behavior of max_depth on a layered model like ResNet"""
        # pyre-fixme[16]: Module `models` has no attribute `resnet18`.
        pretrained_model = models.resnet18(pretrained=True)

        # max_depth = None
        ms1 = get_module_summary(pretrained_model)

        self.assertEqual(len(ms1.submodule_summaries), 10)
        self.assertEqual(len(ms1.submodule_summaries["layer2"].submodule_summaries), 2)
        self.assertEqual(
            len(
                ms1.submodule_summaries["layer2"]
                .submodule_summaries["layer2.0"]
                .submodule_summaries
            ),
            6,
        )

        ms2 = get_module_summary(pretrained_model, max_depth=1)
        self.assertEqual(len(ms2.submodule_summaries), 0)
        self.assertNotIn("layer2", ms2.submodule_summaries)

        ms3 = get_module_summary(pretrained_model, max_depth=2)
        self.assertEqual(len(ms3.submodule_summaries), 10)
        self.assertEqual(len(ms1.submodule_summaries["layer2"].submodule_summaries), 2)
        self.assertNotIn(
            "layer2.0", ms3.submodule_summaries["layer2"].submodule_summaries
        )
        inp = torch.randn(1, 3, 224, 224)
        ms4 = get_module_summary(pretrained_model, max_depth=2, module_input=inp)
        self.assertEqual(len(ms4.submodule_summaries), 10)
        self.assertEqual(ms4.flops_forward, 1814073344)
        self.assertEqual(ms4.flops_backward, 3510132736)
        self.assertEqual(ms4.submodule_summaries["layer2"].flops_forward, 411041792)
        self.assertEqual(ms4.submodule_summaries["layer2"].flops_backward, 822083584)

        # These should stay constant for all max_depth values
        ms_list = [ms1, ms2, ms3, ms4]
        for ms in ms_list:
            self.assertEqual(ms.module_name, "")
            self.assertEqual(ms.module_type, "ResNet")
            self.assertEqual(ms.num_parameters, 11689512)
            self.assertEqual(ms.num_trainable_parameters, 11689512)
            self.assertFalse(ms.has_uninitialized_param)

    def test_module_summary_layer_print(self) -> None:
        model = torch.nn.Conv2d(3, 8, 3)
        ms1 = get_module_summary(model)
        ms2 = get_module_summary(model, max_depth=1)

        summary_table = (
            "Name | Type   | # Params | # Trainable Params | Size in bytes | Contains Uninitialized Param?\n"
            + "---------------------------------------------------------------------------------------------\n"
            + "     | Conv2d | 224      | 224                | 896           | No                           \n"
        )
        self.assertEqual(summary_table, str(ms1))
        self.assertEqual(str(ms1), str(ms2))

    def test_alexnet_print(self) -> None:
        pretrained_model = models.alexnet(pretrained=True)
        ms1 = get_module_summary(pretrained_model, max_depth=1)
        ms2 = get_module_summary(pretrained_model, max_depth=2)
        ms3 = get_module_summary(pretrained_model, max_depth=3)
        ms4 = get_module_summary(pretrained_model)

        summary_table1 = (
            "Name | Type    | # Params | # Trainable Params | Size in bytes | Contains Uninitialized Param?\n"
            + "----------------------------------------------------------------------------------------------\n"
            + "     | AlexNet | 61.1 M   | 61.1 M             | 244 M         | No                           \n"
        )

        summary_table2 = (
            "Name       | Type              | # Params | # Trainable Params | Size in bytes | Contains Uninitialized Param?\n"
            + "--------------------------------------------------------------------------------------------------------------\n"
            + "           | AlexNet           | 61.1 M   | 61.1 M             | 244 M         | No                           \n"
            + "features   | Sequential        | 2.5 M    | 2.5 M              | 9.9 M         | No                           \n"
            + "avgpool    | AdaptiveAvgPool2d | 0        | 0                  | 0             | No                           \n"
            + "classifier | Sequential        | 58.6 M   | 58.6 M             | 234 M         | No                           \n"
        )

        self.assertEqual(summary_table1, str(ms1))
        self.assertEqual(summary_table2, str(ms2))
        self.assertEqual(str(ms3), str(ms4))

    def test_alexnet_print_flops(self) -> None:
        pretrained_model = models.alexnet(pretrained=True)
        inp = torch.randn(1, 3, 224, 224)
        ms1 = get_module_summary(pretrained_model, max_depth=1, module_input=inp)
        ms2 = get_module_summary(pretrained_model, max_depth=2, module_input=inp)
        ms3 = get_module_summary(pretrained_model, max_depth=3, module_input=inp)
        ms4 = get_module_summary(pretrained_model, module_input=inp)

        summary_table1 = (
            "Name | Type    | # Params | # Trainable Params | Size in bytes | Contains Uninitialized Param? | Forward FLOPs | Backward FLOPs\n"
            + "-------------------------------------------------------------------------------------------------------------------------------\n"
            + "     | AlexNet | 61.1 M   | 61.1 M             | 244 M         | No                            | 714 M         | 1.4 G         "
        )

        summary_table2 = (
            "Name       | Type              | # Params | # Trainable Params | Size in bytes | "
            + "Contains Uninitialized Param? | Forward FLOPs | Backward FLOPs\n"
            + "-----------------------------------------------------------------------------------------------------------------------------------------------\n"
            + "           | AlexNet           | 61.1 M   | 61.1 M             "
            + "| 244 M         | No                            | 714 M         | 1.4 G         \n"
            + "features   | Sequential        | 2.5 M    | 2.5 M              "
            + "| 9.9 M         | No                            | 655 M         | 1.2 G         \n"
            + "avgpool    | AdaptiveAvgPool2d | 0        | 0                  "
            + "| 0             | No                            | 0             | 0             \n"
            + "classifier | Sequential        | 58.6 M   | 58.6 M             "
            + "| 234 M         | No                            | 58.6 M        | 117 M         \n"
        )

        self.assertIn(summary_table1, str(ms1))
        self.assertIn(summary_table2, str(ms2))
        self.assertEqual(str(ms3), str(ms4))

    def test_get_human_readable_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "received -1"):
            _get_human_readable_count(-1)
        with self.assertRaisesRegex(TypeError, "received <class 'float'>"):
            # pyre-fixme[6]: For 1st param expected `int` but got `float`.
            _get_human_readable_count(0.1)
        self.assertEqual(_get_human_readable_count(1), "1  ")
        self.assertEqual(_get_human_readable_count(123), "123  ")
        self.assertEqual(_get_human_readable_count(1234), "1.2 K")
        self.assertEqual(_get_human_readable_count(1254), "1.3 K")
        self.assertEqual(_get_human_readable_count(1960), "2.0 K")
        self.assertEqual(_get_human_readable_count(int(1e4)), "10.0 K")
        self.assertEqual(_get_human_readable_count(int(1e6)), "1.0 M")
        self.assertEqual(_get_human_readable_count(int(1e9)), "1.0 B")
        self.assertEqual(_get_human_readable_count(int(1e12)), "1.0 T")
        self.assertEqual(_get_human_readable_count(int(1e15)), "1,000 T")
