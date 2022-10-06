#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import pytorch_sphinx_theme
from torcheval import __version__
import torcheval

from pathlib import Path
import inspect
from types import FunctionType

current_dir = os.path.dirname(__file__)
target_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, target_dir)
print(target_dir)

# -- Project information -----------------------------------------------------

project = "TorchEval"
copyright = "2022, Meta"
author = "Meta"

# The full version, including alpha/beta/rc tags
release = __version__


# -- Link to GitHub Repo -------------------------------------------------
# def linkcode_resolve(domain, info):
#     if domain != 'py':
#         return None
#     if not info['module']:
#         return None
#     filename = info['module'].replace('.', '/')
#     return "https://github.com/pytorch/torcheval/tree/main/torcheval/%s-----%s" % (repr(info), repr(domain))

line_numbers = {}
file_root = Path(inspect.getsourcefile(torcheval)).parent.parent.resolve()


def extract_wrapped(decorated):
    """stolen from here: https://varunver.wordpress.com/2019/03/11/get-source-code-of-a-function-wrapped-by-a-decorator/"""
    closure = (c.cell_contents for c in decorated.__closure__)
    return next((c for c in closure if isinstance(c, FunctionType)), None)

def autodoc_process_docstring(app, what, name, obj, options, lines):
    """We misuse this autodoc hook to get the file names & line numbers because we have access
    to the actual object here.
    
    Stolen from: https://github.com/sphinx-doc/sphinx/issues/1556
    """

    try:
        source_lines, start_line = inspect.getsourcelines(obj)
        end_line = start_line + len(source_lines)
        if "functional" in name:
            file = str(Path(inspect.getfile(extract_wrapped(obj))).relative_to(file_root))
        else:
            file = str(Path(inspect.getsourcefile(obj)).relative_to(file_root))
        line_numbers[name] = (file, start_line, end_line)
    except Exception:
        pass


def build_url(file_line_info):
    filepath, line_start, line_end = file_line_info
    return f"https://github.com/pytorch/torcheval/blob/main/{filepath}#L{line_start}-L{line_end}"


def linkcode_resolve(_, info):
    """See www.sphinx-doc.org/en/master/usage/extensions/linkcode.html
    
    Stolen from: https://github.com/sphinx-doc/sphinx/issues/1556"""
    combined = '.'.join((info['module'], info['fullname']))
    line_info = line_numbers.get(combined)
    
    if not line_info:
        # Try the __init__
        line_info = line_numbers.get(f"{combined.rsplit('.', 1)[0]}.__init__")
    if not line_info:
        # Try the class
        line_info = line_numbers.get(f"{combined.rsplit('.', 1)[0]}")
    if not line_info:
        # Try the module
        line_info = line_numbers.get(info['module'])

    if not line_info:
        return
    
    return build_url(line_info)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.linkcode",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pytorch_sphinx_theme"
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# where to find external docs
intersphinx_mapping = {
    "torch": ("https://pytorch.org/docs/stable/", None),
}

def setup(app):
    app.connect("autodoc-process-docstring", autodoc_process_docstring)