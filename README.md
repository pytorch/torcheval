# TorchEval

<p align="center">
<a href="https://github.com/pytorch-labs/torcheval/actions?query=branch%3Amain"><img src="https://img.shields.io/github/workflow/status/pytorch-labs/torcheval/unit%20test/main" alt="build status"></a>
<a href="https://pypi.org/project/torcheval"><img src="https://img.shields.io/pypi/v/torcheval" alt="pypi version"></a>
<a href="https://pypi.org/project/torcheval-nightly"><img src="https://img.shields.io/pypi/v/torcheval-nightly?label=nightly" alt="pypi nightly version"></a>
<a href="https://github.com/pytorch-labs/torcheval/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/torcheval" alt="bsd license"></a>
</div>
<a href="https://pytorch-labs.github.io/torcheval"><img src="https://img.shields.io/badge/docs-main-brightgreen" alt="docs"></a>
<p>

**This library is currently in Alpha and currently does not have a stable release. The API may change and may not be backward compatible. If you have suggestions for improvements, please open a GitHub issue. We'd love to hear your feedback.**

A library that contains a rich collection of performant PyTorch model metrics, a simple interface to create new metrics, a toolkit to facilitate metric computation in distributed training and tools for PyTorch model evaluations.

## Installing TorchEval
Requires Python >= 3.7 and PyTorch >= 1.11

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
git clone https://github.com/pytorch-labs/torcheval
cd torcheval
pip install -r requirements.txt
python setup.py install
```

## Quick Start

```bash
cd torcheval
python examples/simple_example.py
```

## Documentation

Documentation can be found at at [pytorch-labs.github.io/torcheval](https://pytorch-labs.github.io/torcheval)

## Using TorchEval

TorchEval can be run on CPU, GPU, and Multi-GPUs/Muti-Nodes.

For the multiple devices usage:
```python
import torch
from torcheval.metrics.toolkit import sync_and_compute
from torcheval.metrics import MulticlassAccuracy

local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])
world_size  = int(os.environ["WORLD_SIZE"])

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

        # metric.update() updates the metric state with new data
        metric.update(input, target)


        # metric.compute() returns metric value from all seen data on the local process.
        local_compute_result = metric.compute()

        # sync_and_compute(metric) returns metric value from all seen data on all processes.
        # It gives the same result as ``metric.compute()`` if it's run on single process.
        global_compute_result = sync_and_compute(metric)

        # The final result is collected by rank 0
        if global_rank == 0:
            print(global_compute_result)

    # metric.reset() cleans up all seen data
    metric.reset()
```
See the [example directory](https://github.com/pytorch-labs/torcheval/tree/main/examples) for more examples.

## Contributing
We welcome PRs! See the [CONTRIBUTING](CONTRIBUTING.md) file.

## License
TorchEval is BSD licensed, as found in the [LICENSE](LICENSE) file.
