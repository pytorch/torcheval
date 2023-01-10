#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This script shows how to build the docs.
# 1. First ensure you have the installed requirements for torcheval in `requirements.txt`
# 2. Then make sure you have installed the requirements inside `docs/requirements.txt`
# 3. Finally cd into docs/ and source this script. Sphinx reads through the installed module
# pull docstrings, so this script just installs the current version of torcheval on your
# system before it builds the docs with `make html`
cd .. || exit
pip install --no-build-isolation .
cd docs || exit
make html
