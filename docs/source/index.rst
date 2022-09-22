TorchEval
===========================================

A library with tooling for model evalutations. At a high level TorchEval:

1. Contains a rich collection of optimized metric calulcations
2. Integrates seemlessly with distributed training and tools using `torch.distributed <https://pytorch.org/tutorials/beginner/dist_overview.html>`_
3. Provides an easy to use interface for adding custom metrics
4. Provides tools for profiling memory and compute requirements for PyTorch based models

TorchEval provides some of the same functionality as `TorchMetrics` and ``sklearn.metrics``, but focuses on performant calculations and utilizing vectorization and GPU acceleration where possible via PyTorch. Although TorchEval was built specifically to integrate with PyTorch training loops, it can be used as a standalone library for models built with other frameworks.


QuickStart
===========================================

Installing
-----------------

TorchEval can be installed from PyPi via

.. code-block:: console

   pip install torcheval

or from github

.. code-block:: console

   git clone https://github.com/pytorch-labs/torcheval
   cd torcheval
   pip install -r requirements.txt
   python setup.py install

Usage
-----------------

TorchEval provides two interfaces to each metric. If you are working in a single process environment, it is simplest to use metrics from the ``functional`` submodule. These can be found in ``torcheval.metrics.functional``.

.. code-block:: python

   from torcheval.metrics.functional import binary_f1_score
   predictions = model(inputs)
   f1_score = binary_f1_score(predictions, targets)

We can use the same metric in the class based route, which is more appropriate for multi-process settings

.. code-block:: python

   from torcheval.metrics import BinaryF1Score
   predictions = model(inputs)
   metric = BinaryF1Score()
   metric.update(predictions, targets)
   f1_score = metric.compute()

Read more about the class based method in the distributed example.

Further Reading
-----------------
* Check out the guides exaplaining the compute example
* Check out the distributed example
* Check out how to make your own metric

TorchEval API
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   torcheval.metrics.rst
   torcheval.metrics.functional.rst
   torcheval.tools.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
