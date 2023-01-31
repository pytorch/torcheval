.. currentmodule:: torcheval

Use Metrics in TorchEval
========================

PyTorch evaluation metrics are one of the core offerings of TorchEval.
For most metrics, we offer both stateful class-based interfaces that only accumulate necessary data until told to compute the metric, and pure functional interfaces.


Class Metrics
--------------

The class metrics keeps track of metric states, which enables them to be able to calculate values through accumulations and
synchronizations across multiple processes. The base class is :obj:`torcheval.metrics.Metric`.

The core APIs of class metrics are ``update()``, ``compute()`` and ``reset()``.

- ``update()``: Update the metric states with input data. This is often used when new data needs to be added for metric computation.
- ``compute()``: Compute the metric values from the metric state, which are updated by previous ``update()`` calls. The compute frequency can be less than the update frequency.
- ``reset()``: Reset the metric state variables to their default value. Usually this is called at the end of every epoch to clean up metric states.

.. note::

    Class metrics keep track of internal states that are updated by input data passed to ``update()`` calls. This means that metric states should be moved to the
    same device as the input data. You can directly pass in device on initialization or use the ``to(device)`` API. The ``.device`` property shows the device of the metric states.

Below is an example of using class metric in a simple training script.

.. code-block:: python

    import torch
    from torcheval.metrics import MulticlassAccuracy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = MulticlassAccuracy(device=device)
    num_epochs, num_batches, batch_size = 4, 8, 10
    num_classes = 3

    # number of batches between metric computations
    compute_frequency = 2

    for epoch in range(num_epochs):
        for batch_idx in range(num_batches):
            input = torch.randint(high=num_classes, size=(batch_size,), device=device)
            target = torch.randint(high=num_classes, size=(batch_size,), device=device)

            # metric.update() updates the metric state with new data
            metric.update(input, target)

            if (batch_idx + 1) % compute_frequency == 0:
                    print(
                        "Epoch {}/{}, Batch {}/{} --- acc: {:.4f}".format(
                            epoch + 1,
                            num_epochs,
                            batch_idx + 1,
                            num_batches,
                            # metric.compute() returns metric value from all seen data
                            metric.compute(),
                        )
                    )

        # metric.reset() reset metric states. It's typically called after the epoch completes.
        metric.reset()

Save and Load Metrics
^^^^^^^^^^^^^^^^^^^^^

Class metrics also implements the stateful protocol, ``.state_dict()`` and ``.load_state_dict()``. Those functions can be used to save and load metrics.

.. code-block:: python

    import torch
    from torcheval.metrics import MulticlassAccuracy

    metric = MulticlassAccuracy()
    input = torch.tensor([0, 2, 1, 3])
    target = torch.tensor([0, 1, 2, 3])
    metric.update(input, target)

    state_dict = metric.state_dict()
    loaded_metric = MulticlassAccuracy()
    loaded_metric.load_state_dict(state_dict)

    # returns torch.tensor(0.5)
    loaded_metric.compute()


Functional Metrics
------------------


Functional metrics are simple python functions that calculate the metric value from input data. They are light-weighted and relatively faster since they don't need to keep and operate on metric states.
The example below shows calculating metric value with the functional version.

.. code-block:: python

    import torch
    from torcheval.metrics.functional import multiclass_accuracy

    input = torch.tensor([0, 2, 1, 3])
    target = torch.tensor([0, 1, 2, 3])
    # returns torch.tensor(0.5)
    multiclass_accuracy(input, target)
