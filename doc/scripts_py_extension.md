# Shared Interfaces and Extension Guide

## Shared Interfaces

The training and prediction code no longer instantiate `ComplexRangeCNN`
directly. Concrete model implementations live in separate files such as
`network/models/model_complex_cnn_range.py` and
`network/models/model_real_cnn_range.py`. New models should be added through the
registry in `network/model.py`:

```python
MODEL_REGISTRY = {
    "complex_cnn_range": (ComplexRangeCNNConfig, ComplexRangeCNN),
    "real_cnn_range": (RealRangeCNNConfig, RealRangeCNN),
    "resnet18_range": (ResNetRangeConfig, ResNet18Range),
    "resnet50_range": (ResNetRangeConfig, ResNet50Range),
}
```

The shared helpers are:

- `build_model_config(model_name, input_shape, ...)`: creates the config for a
  selected model
- `build_model(model_name, config)`: constructs the PyTorch module
- `model_config_from_checkpoint(checkpoint)`: restores the model type and config
  for resume and prediction
- `serialize_model_config(config)`: stores config data in checkpoint files

Data preparation is centralized in `network/data.py`:

- `RbdRangeH5Dataset`: lazy HDF5 dataset
- `random_split_bundle(...)`: one dataset with seeded random train/validation
  split
- `fixed_split_bundle(...)`: separate training and validation HDF5 inputs
- `DatasetBundle`: common object consumed by the training loop

This means new model architectures or new split strategies should plug into the
registry or return a `DatasetBundle`, while the epoch loop, checkpoint format,
resume logic, prediction CSV, and plot generation remain shared.

## Change Model

The public training and prediction code should stay model-agnostic:

```text
Network_main.py
  -> network.cli
      -> network.training
          -> network.model
          -> network.data
      -> network.prediction
          -> network.model
          -> network.data
```

To add a new model architecture:

1. Create a new file under `scripts_py/network/models/`, for example:

```text
scripts_py/network/models/model_real_cnn_range.py
```

2. Define a dataclass config and a PyTorch module. The config must include
   `input_elements` and `input_freq_bins`, because `network/model.py` fills
   those fields from the HDF5 input shape:

```python
from dataclasses import dataclass

from torch import nn


@dataclass
class ModelConfig:
    input_elements: int
    input_freq_bins: int
    hidden_channels: int = 32
    dropout: float = 0.15


class RealRangeCNN(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        ...

    def forward(self, x):
        ...
```

The forward output should have shape `[batch, 1]` and should predict normalized
range, not km. The shared training loop handles target normalization and metric
conversion back to km.

3. Register the model in `scripts_py/network/model.py`:

```python
from .models.model_real_cnn_range import (
    RealRangeCNN,
    ModelConfig as RealRangeCNNConfig,
)


MODEL_REGISTRY = {
    "complex_cnn_range": (ComplexRangeCNNConfig, ComplexRangeCNN),
    "real_cnn_range": (RealRangeCNNConfig, RealRangeCNN),
}
```

4. Add the model name to the `--model` choices in `scripts_py/network/cli.py`.

5. Train the new model:

```powershell
python scripts_py/Network_main.py train `
  --model real_cnn_range `
  --data periodic_4_1 `
  --no-resume
```

Use `--no-resume` when switching architectures. Resume checkpoints store
`model_name` and `model_config`; prediction checks that the checkpoint model
matches the `--model` argument, so mismatched model/checkpoint combinations fail
fast.

Prediction also requires the model name. If `--checkpoint` is omitted, the
checkpoint is resolved from that model's `train_outputs` directory:

```powershell
python scripts_py/Network_main.py predict `
  --model real_cnn_range `
  --checkpoint outputs/networks_results/real_cnn_range/train_outputs/periodic_4_1_best_0426_102042.pt `
  --data periodic_4_1
```

The checkpoint records `model_name` and `dataset_name`, so `predict` rebuilds
the correct registered model automatically and checks that it matches the
command arguments.
