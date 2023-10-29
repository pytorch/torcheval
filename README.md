# TorchEval

<p align="center">
<a href="https://github.com/pytorch/torcheval/actions?query=branch%3Amain"><img src="https://img.shields.io/github/actions/workflow/status/pytorch/torcheval/.github/workflows/unit_test.yaml?branch=main" alt="build status"></a>
<a href="https://pypi.org/project/torcheval"><img src="https://img.shields.io/pypi/v/torcheval" alt="pypi version"></a>
<a href="https://pypi.org/project/torcheval-nightly"><img src="https://img.shields.io/pypi/v/torcheval-nightly?label=nightly" alt="pypi nightly version"></a>
<a href="https://github.com/pytorch/torcheval/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/torcheval" alt="bsd license"></a>
</div>
<a href="https://pytorch.github.io/torcheval"><img src="https://img.shields.io/badge/docs-main-brightgreen" alt="docs"></a>
<p>

**This library is currently in Alpha and currently does not have a stable release. The API may change and may not be backward compatible. If you have suggestions for improvements, please open a GitHub issue. We'd love to hear your feedback.**

A library that contains a rich collection of performant PyTorch model metrics, a simple interface to create new metrics, a toolkit to facilitate metric computation in distributed training and tools for PyTorch model evaluations.

## Installing TorchEval
Requires Python >= 3.8 and PyTorch >= 1.11

From pip:

```bash
pip install torcheval
```

For nighly build version
```bash
pip install --pre torcheval-nightly
```

From source:

```bash
git clone https://github.com/pytorch/torcheval
cd torcheval
pip install -r requirements.txt
python setup.py install
```

## Quick Start

Take a look at the [quickstart notebook](https://github.com/pytorch/torcheval/blob/main/examples/Introducing_TorchEval.ipynb), or fork it on [Colab](https://colab.research.google.com/github/pytorch/torcheval/blob/main/examples/Introducing_TorchEval.ipynb).

There are more examples in the [examples](https://github.com/pytorch/torcheval/blob/main/examples) directory:

```bash
cd torcheval
python examples/simple_example.py
```

## Documentation

Documentation can be found at at [pytorch.org/torcheval](https://pytorch.org/torcheval)

## Using TorchEval

TorchEval can be run on CPU, GPU, and in a multi-process or multi-GPU setting. Metrics are provided in two interfaces, functional and class based. The functional interfaces can be found in `torcheval.metrics.functional` and are useful when your program runs in a single process setting. To use multi-process or multi-gpu configurations, the class-based interfaces, found in `torcheval.metrics` provide a much simpler experience. The class based interfaces also allow you to defer some of the computation of the metric by calling `update()` multiple times before `compute()`. This can be advantageous even in a single process setting due to saved computation overhead.

### Single Process
For use in a single process program, the simplest use case utilizes a functional metric. We simply import the metric function and feed in our outputs and targets. The example below shows a minimal PyTorch training loop that evaluates the multiclass accuracy of every fourth batch of data.

#### Functional Version (immediate computation of metric)
```python
import torch
from torcheval.metrics.functional import multiclass_accuracy

NUM_BATCHES = 16
BATCH_SIZE = 8
INPUT_SIZE = 10
NUM_CLASSES = 6
eval_frequency = 4

model = torch.nn.Sequential(torch.nn.Linear(INPUT_SIZE, NUM_CLASSES), torch.nn.ReLU())
optim = torch.optim.Adagrad(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

metric_history = []
for batch in range(NUM_BATCHES):
    input = torch.rand(size=(BATCH_SIZE, INPUT_SIZE))
    target = torch.randint(size=(BATCH_SIZE,), high=NUM_CLASSES)
    outputs = model(input)

    loss = loss_fn(outputs, target)
    optim.zero_grad()
    loss.backward()
    optim.step()

    # metric only computed every 4 batches,
    # data from previous three batches is lost
    if (batch + 1) % eval_frequency == 0:
        metric_history.append(multiclass_accuracy(outputs, target))
```
### Single Process with Deferred Computation

#### Class Version (enables deferred computation of metric)
```python
import torch
from torcheval.metrics import MulticlassAccuracy

NUM_BATCHES = 16
BATCH_SIZE = 8
INPUT_SIZE = 10
NUM_CLASSES = 6
eval_frequency = 4

model = torch.nn.Sequential(torch.nn.Linear(INPUT_SIZE, NUM_CLASSES), torch.nn.ReLU())
optim = torch.optim.Adagrad(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()
metric = MulticlassAccuracy()

metric_history = []
for batch in range(NUM_BATCHES):
    input = torch.rand(size=(BATCH_SIZE, INPUT_SIZE))
    target = torch.randint(size=(BATCH_SIZE,), high=NUM_CLASSES)
    outputs = model(input)

    loss = loss_fn(outputs, target)
    optim.zero_grad()
    loss.backward()
    optim.step()

    # metric only computed every 4 batches,
    # data from previous three batches is included
    metric.update(input, target)
    if (batch + 1) % eval_frequency == 0:
        metric_history.append(metric.compute())
        # remove old data so that the next call
        # to compute is only based off next 4 batches
        metric.reset()
```

### Multi-Process or Multi-GPU
For usage on multiple devices a minimal example is given below. In the normal `torch.distributed` paradigm, each device is allocated its own process gets a unique numerical ID called a "global rank", counting up from 0.

#### Class Version (enables deferred computation and multi-processing)
```python
import torch
from torcheval.metrics.toolkit import sync_and_compute
from torcheval.metrics import MulticlassAccuracy

# Using torch.distributed
local_rank = int(os.environ["LOCAL_RANK"]) #rank on local machine, i.e. unique ID within a machine
global_rank = int(os.environ["RANK"]) #rank in global pool, i.e. unique ID within the entire process group
world_size  = int(os.environ["WORLD_SIZE"]) #total number of processes or "ranks" in the entire process group

device = torch.device(
    f"cuda:{local_rank}"
    if torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    else "cpu"
)

metric = MulticlassAccuracy(device=device)
num_epochs, num_batches = 4, 8

for epoch in range(num_epochs):
    for i in range(num_batches):
        input = torch.randint(high=5, size=(10,), device=device)
        target = torch.randint(high=5, size=(10,), device=device)

        # Add data to metric locally
        metric.update(input, target)

        # metric.compute() will returns metric value from
        # all seen data on the local process since last reset()
        local_compute_result = metric.compute()

        # sync_and_compute(metric) syncs metric data across all ranks and computes the metric value
        global_compute_result = sync_and_compute(metric)
        if global_rank == 0:
            print(global_compute_result)

    # metric.reset() clears the data on each process so that subsequent
    # calls to compute() only act on new data
    metric.reset()
```
See the [example directory](https://github.com/pytorch/torcheval/tree/main/examples) for more examples.

## Contributing
We welcome PRs! See the [CONTRIBUTING](CONTRIBUTING.md) file.

## License
TorchEval is BSD licensed, as found in the [LICENSE](LICENSE) file.
