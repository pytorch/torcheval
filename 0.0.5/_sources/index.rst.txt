TorchEval
===========================================

A library with simple and straightforward tooling for model evaluations and a delightful user experience. At a high level TorchEval:

1. Contains a rich collection of high performance metric calculations out of the box. We utilize vectorization and GPU acceleration where possible via PyTorch.
2. Integrates seemlessly with distributed training and tools using `torch.distributed <https://pytorch.org/tutorials/beginner/dist_overview.html>`_
3. Is designed with extensibility in mind: you have the freedom to easily create your own metrics and leverage our toolkit.
4. Provides tools for profiling memory and compute requirements for PyTorch based models.

TorchEval Tutorials
-------------------
.. toctree::
   :maxdepth: 2
   :caption: Examples:

   QuickStart Notebook <https://github.com/pytorch/torcheval/blob/main/examples/Introducing_TorchEval.ipynb>
   metric_example.rst

QuickStart
===========================================

Installing
-----------------

TorchEval can be installed from PyPi via

.. code-block:: console

   pip install torcheval

or from github

.. code-block:: console

   git clone https://github.com/pytorch/torcheval
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

We can use the same metric in the class based route, which provides tools that make computation simple in a multi-process setting. On a single device, you can use the class based metrics as follows:

.. code-block:: python

   from torcheval.metrics import BinaryF1Score
   predictions = model(inputs)
   metric = BinaryF1Score()
   metric.update(predictions, targets)
   f1_score = metric.compute()

In a multi-process setting, the data from each process must be synchronized to compute the metric across the full dataset. To do this, simply replace ``metric.compute()`` with ``sync_and_compute(metric)``:

.. code-block:: python

   from torcheval.metrics import BinaryF1Score
   from torcheval.metrics.toolkit import sync_and_compute
   predictions = model(inputs)
   metric = BinaryF1Score()
   metric.update(predictions, targets)
   f1_score = sync_and_compute(metric)

Read more about the class based method in the distributed example.

Further Reading
-----------------
* Check out the guides explaining the compute example
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
