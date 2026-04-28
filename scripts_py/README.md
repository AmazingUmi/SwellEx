# Complex CNN Range Regression

This folder contains a first PyTorch implementation for source-to-VLA range
regression from the HDF5 files written by
`scripts_matlab/Signals_Segmentation.m`.

## Code Layout

The command you run is still:

```text
scripts_py/Network_main.py
```

That file is now a small compatibility entry point. The implementation is split
by function under `scripts_py/network/`:

- `cli.py`: command-line parser and `train`/`predict` subcommands
- `data.py`: HDF5 path resolution, layout detection, lazy dataset loading, and
  reusable train/validation split bundles
- `model.py`: model registry and construction helpers used by training, resume,
  and prediction
- `models/model_complex_cnn_range.py`: complex convolution layers and
  `ComplexRangeCNN`
- `models/model_real_cnn_range.py`: real convolution model fed by normalized
  complex magnitude
- `training.py`: target normalization, metrics, checkpoint resume, and the
  shared training loop
- `prediction.py`: model-registry checkpoint loading, CSV writing, and
  prediction plot output
- `paths.py`: shared project output/input defaults

## HDF5 Input

The training script expects:

- `outputs/Datasets/<split_strategy>/*_train.h5`: training source data
- `outputs/Datasets/<split_strategy>/*_test.h5`: held-out test data
- `outputs/Datasets/<split_strategy>/*_metadata.json`: dataset split metadata
- `/X`: `[window, element, frequency, 2]`
- `/X[..., 0]`: real part of `green_freq`
- `/X[..., 1]`: imaginary part of `green_freq`
- `/y_range_km`: source-to-array range label in km
- `/valid_sample`: optional mask; only finite valid labels are used
- `/split/source_segment_idx`: original global segment index in the full record

The dataset is loaded lazily with `h5py`, so samples are read on demand instead
of loading the whole HDF5 file into memory.

MATLAB-created HDF5 files may appear in Python as
`[2, frequency, element, window]` because of dimension ordering. The loader
detects this automatically and converts each sample to PyTorch layout
`[2, element, frequency]`.

## MATLAB Split Strategies

`scripts_matlab/Signals_Segmentation.m` writes datasets under
`outputs/Datasets/<split_strategy>/`. The split strategy is selected in the
MATLAB user-parameter section:

```matlab
split_strategy = "periodic";

switch split_strategy
    case "periodic"
        split_options = struct();
        split_options.train_test_ratio = [4 1];
    case "Range_nearby"
        split_options = struct();
        split_options.half_duration_s = 300;
        split_options.gap_s = 30;
        split_options.train_side = "before";
end
```

Available strategies:

- `periodic`: deterministic interleaving controlled by `train_test_ratio`, for
  example `periodic_4_1`.
- `Range_nearby`: find the minimum source range, take symmetric time windows on
  both sides, and use one side for training and the other for testing. Configure
  with:

```matlab
split_strategy = "Range_nearby";
split_options.half_duration_s = 300;
split_options.gap_s = 30;
split_options.train_side = "before";  % or "after"
```

`Range_nearby` is intended to test generalization between the inbound and
outbound portions of the run while keeping the range neighborhood comparable.
For this strategy, the HDF5 filename uses the actual selected symmetric time
range around the minimum range point. The full candidate segmentation interval
is still recorded in `*_metadata.json` as `candidate_segment_start_s` and
`candidate_segment_end_s`.

## Install

Use a Python environment with PyTorch, NumPy, h5py, and Matplotlib:

```powershell
pip install torch numpy h5py matplotlib
```

Pick the PyTorch install command that matches your CUDA version if you want GPU
training.

### HaiQin1 Local Toolchain

On the local machine identified as `HaiQin1`, use these explicit executables for
Codex-run debugging and smoke tests.

Python/PyTorch:

```text
G:\software\Anaconda\envs\pytorch\python.exe
```

MATLAB:

```text
C:\Program Files\MATLAB\R2025b\bin\matlab.exe
```

Quick Python environment check:

```powershell
G:\software\Anaconda\envs\pytorch\python.exe -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
```

When Codex runs local Python tests on `HaiQin1`, prefer this interpreter instead
of the default `python`, for example:

```powershell
G:\software\Anaconda\envs\pytorch\python.exe -m py_compile scripts_py\Network_main.py scripts_py\network\training.py
```

When Codex runs local MATLAB checks on `HaiQin1`, prefer the MATLAB executable
above instead of the default `matlab`, for example:

```powershell
& "C:\Program Files\MATLAB\R2025b\bin\matlab.exe" -batch "disp('Signals_Segmentation'); checkcode('scripts_matlab/Signals_Segmentation.m');"
```

## Train

From the project root, explicitly pass a model and a dataset:

```powershell
python scripts_py/Network_main.py train `
  --model complex_cnn_range `
  --data periodic_4_1
```

The command above resolves the dataset directory
`outputs/Datasets/periodic_4_1`, loads `*_train.h5`, uses `*_val.h5` when
present, and otherwise splits the training files 3:1 into training and
validation sets:

- source: `outputs/Datasets/periodic_4_1/*_train.h5`
- train/val split: 75% / 25%

Useful options:

```powershell
python scripts_py/Network_main.py train `
  --model complex_cnn_range `
  --data periodic_4_1 `
  --val-fraction 0.25 `
  --epochs 200 `
  --batch-size 16 `
  --lr 3e-4 `
  --output-dir outputs/networks_results
```

For custom HDF5 globs, still pass the dataset code through `--data` and override
the training files with `--train-data`:

```powershell
python scripts_py/Network_main.py train `
  --model complex_cnn_range `
  --data periodic_4_1 `
  --train-data "outputs/Datasets/periodic_4_1/*_train.h5"
```

Use `--model complex_cnn_range` to select the model architecture:

```powershell
python scripts_py/Network_main.py train `
  --model complex_cnn_range `
  --data periodic_4_1 `
  --no-resume
```

The built-in model choices are:

- `complex_cnn_range`: complex-valued convolution model using real/imaginary
  input channels
- `real_cnn_range`: real-valued convolution model that converts real/imaginary
  input to magnitude, then normalizes the magnitude map inside the model

To train the real-valued magnitude model:

```powershell
python scripts_py/Network_main.py train `
  --model real_cnn_range `
  --data periodic_4_1 `
  --no-resume
```

To use a fixed validation split instead of a random split from the training
files, pass both `--train-data` and `--val-data`:

```powershell
python scripts_py/Network_main.py train `
  --model complex_cnn_range `
  --data periodic_4_1 `
  --no-resume
```

Outputs:

- `outputs/networks_results/<model_name>/train_outputs/<dataset>_best_MMDD_HHMMSS.pt`
- `outputs/networks_results/<model_name>/train_outputs/<dataset>_last_MMDD_HHMMSS.pt`
- `outputs/networks_results/<model_name>/train_outputs/<dataset>_history_MMDD_HHMMSS.json`

For `complex_cnn_range` trained on `periodic_4_1`, these are:

- `outputs/networks_results/complex_cnn_range/train_outputs/periodic_4_1_best_MMDD_HHMMSS.pt`
- `outputs/networks_results/complex_cnn_range/train_outputs/periodic_4_1_last_MMDD_HHMMSS.pt`
- `outputs/networks_results/complex_cnn_range/train_outputs/periodic_4_1_history_MMDD_HHMMSS.json`

`--output-dir` is the base output directory. With the default `--output-dir
outputs/networks_results`, each model gets its own result folder:

```text
outputs/
  networks_results/
    complex_cnn_range/
      train_outputs/
      test_outputs/
```

If `--output-dir` already names the model folder, for example
`--output-dir outputs/networks_results/complex_cnn_range`, the script uses that
folder directly and does not create
`outputs/networks_results/complex_cnn_range/complex_cnn_range`.

When a previous `<dataset>_last_MMDD_HHMMSS.pt` exists under
`outputs/networks_results/<model_name>/train_outputs`, `train`
automatically continues from the newest checkpoint for the selected `--model`
and `--data`. In this case `--epochs` is the number of additional epochs to run.
Use `--no-resume` to force a fresh run, or `--resume-checkpoint
path/to/checkpoint.pt` to continue from a specific file. New checkpoints store
`dataset_name`; resume fails if the checkpoint dataset does not match `--data`.

Examples:

```powershell
# Continue from the newest saved last checkpoint for 50 more epochs.
python scripts_py/Network_main.py train `
  --model complex_cnn_range `
  --data periodic_4_1 `
  --epochs 50

# Start a new run and ignore previous checkpoints.
python scripts_py/Network_main.py train `
  --model complex_cnn_range `
  --data periodic_4_1 `
  --no-resume

# Continue from a specific checkpoint.
python scripts_py/Network_main.py train `
  --model complex_cnn_range `
  --data periodic_4_1 `
  --resume-checkpoint outputs/networks_results/complex_cnn_range/train_outputs/periodic_4_1_last_0426_102042.pt `
  --epochs 50
```

Each checkpoint stores enough state to resume training:

- `model_name` and `model_config`
- model weights
- optimizer and scheduler state
- target normalization statistics
- train/validation split metadata
- full training history

## Predict

```powershell
python scripts_py/Network_main.py predict `
  --model complex_cnn_range `
  --data periodic_4_1
```

If `--checkpoint` is omitted, prediction first tries
`outputs/networks_results/<model_name>/train_outputs/<dataset>_best.pt`. If
that fixed path does not exist, the newest
`<dataset>_best_MMDD_HHMMSS.pt` file in the model's `train_outputs` directory is
used automatically.

The checkpoint contains `model_name` and, for new checkpoints, `dataset_name`,
so prediction reconstructs the matching registered model and fails fast if the
checkpoint does not match `--model` or `--data`.

Older checkpoints without the dataset prefix, such as `best_0426_162801.pt`,
are not used by automatic lookup. Pass them explicitly with `--checkpoint` if
they are still needed.

Predictions are written to:

```text
outputs/networks_results/complex_cnn_range/test_outputs/periodic_4_1_predictions_MMDD_HHMMSS.csv
```

A plot is also written to:

```text
outputs/networks_results/complex_cnn_range/test_outputs/periodic_4_1_range_prediction_MMDD_HHMMSS.png
```

The prediction CSV includes `source_segment_idx` when the HDF5 file provides
`/split/source_segment_idx`, so rows can be mapped back to MATLAB's global
segment numbering.

The plot draws the label range as a connected line over `window_center_s` and
draws the predicted ranges as scatter points. Use `--no-plot` to skip plot
generation, or `--plot-path` to choose another output file.

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

4. Add the model name to the `--model` choices in `scripts_py/network/cli.py`:

```python
train_parser.add_argument(
    "--model",
    required=True,
    choices=["complex_cnn_range", "real_cnn_range", "new_model_name"],
    help="Model architecture to train.",
)
```

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

### Built-in Real Magnitude Model

`real_cnn_range` receives the same dataset tensor shape as the complex model:

```text
x: [batch, 2, element, frequency]
```

Inside `model_real_cnn_range.py`, it computes:

```text
mag = sqrt(real^2 + imag^2 + 1e-8)
mag_norm = (mag - sample_mean) / sample_std
```

Then `mag_norm` is passed to a real-valued CNN. Because this model performs
magnitude normalization internally, the training pipeline automatically disables
the dataset-level real/imaginary channel normalization for `real_cnn_range`.

## Change Dataset

The HDF5 dataset contract is:

```text
/X                         required, 4-D, one real/imag axis of length 2
/y_range_km                required, range labels in km
/valid_sample              optional, boolean mask
/split/source_segment_idx  optional, original segment id
/time/window_center_s      optional, plotting x-axis
```

The loader accepts either Python-style layout:

```text
[window, element, frequency, 2]
```

or MATLAB-imported layouts where the axes appear reordered, as long as it can
identify the sample axis, real/imag axis, element axis, and frequency axis.

### Random Split From Training Files

Use this mode when one or more files should be merged and split by `--seed`:

```powershell
python scripts_py/Network_main.py train `
  --model complex_cnn_range `
  --data periodic_4_1 `
  --val-fraction 0.25 `
  --no-resume
```

The files are merged into one dataset and split by `--seed`. The checkpoint
stores `train_indices`, `val_indices`, `val_fraction`, and `split_mode`.

### Fixed Train/Validation Split

Use this when MATLAB or another preprocessing step already produced separate
training and validation files:

```powershell
python scripts_py/Network_main.py train `
  --model complex_cnn_range `
  --data periodic_4_1 `
  --no-resume
```

In this mode no random validation split is applied. Training and validation
files must share the same input shape `[2, element, frequency]`.

### Override Training Files

Use `--train-data` when the dataset code should still identify the experiment,
but the training files need to be selected manually:

```powershell
python scripts_py/Network_main.py train `
  --model complex_cnn_range `
  --data periodic_4_1 `
  --train-data "outputs/Datasets/periodic_4_1/*_train.h5" `
  --val-fraction 0.25 `
  --no-resume
```

If `--val-data` is omitted, the selected training files are randomly split by
`--seed` and `--val-fraction`.

### Predict On A Different Test Set

Prediction data only needs to match the model input shape and HDF5 field names:

```powershell
python scripts_py/Network_main.py predict `
  --model complex_cnn_range `
  --data periodic_4_1
```

The default fixed checkpoint path is a convenience alias. If it does not exist,
the newest matching `best_*.pt` file in the model's `train_outputs` directory is
used automatically.

### Add A New Dataset Format

If a future dataset cannot be represented by `RbdRangeH5Dataset`, add a new
dataset class in `network/data.py` or a new data module, then return the same
`DatasetBundle` fields expected by `network/training.py`:

- `train_dataset`
- `val_dataset`
- `train_idx`
- `val_idx`
- `train_labels`
- `input_shape`
- `split_mode`
- `source_paths`

Keeping the `DatasetBundle` interface stable lets the shared training loop,
checkpoint format, resume logic, and prediction outputs stay unchanged.

## Model Notes

The network uses explicit complex convolutions:

```text
(Wr + jWi) * (xr + jxi)
  = (Wr*xr - Wi*xi) + j(Wr*xi + Wi*xr)
```

Complex feature maps pass through batch normalization and a magnitude-gated
activation. Before the final MLP head, real and imaginary maps are converted to
magnitude features for scalar range regression. Targets are normalized with the
training-set mean and standard deviation, while reported metrics are in km.

## Network Architecture

The model implementation is in `network/models/model_complex_cnn_range.py`:

- `ComplexRangeCNN`: full range-regression model
- `ComplexConvBlock`: complex convolution + complex batch normalization + modReLU
- `ComplexConv2d`: complex convolution represented by two real `Conv2d` layers
- `ComplexBatchNorm2d`: independent batch normalization for real and imaginary maps
- `ComplexModReLU`: magnitude-gated activation that preserves complex phase

Input samples are loaded as real and imaginary channels:

```text
x: [batch, 2, element, frequency]
   channel 0 = real
   channel 1 = imaginary
```

The complex layers treat this as one complex input channel. Each complex feature
channel is stored as two real tensor channels, so an internal complex width of
`C` appears in PyTorch tensors as `2*C` channels.

With the default `--base-channels 16`, the feature extractor is:

```text
Input: [B, 2, E, F]

Block 1: ComplexConv2d  1 -> 16, kernel 3x3, stride (1, 2), padding 1
         ComplexBatchNorm2d(16)
         ComplexModReLU(16)
         Output complex shape: [B, 16, E, ceil(F/2)]

Block 2: ComplexConv2d 16 -> 32, kernel 3x3, stride (2, 2), padding 1
         ComplexBatchNorm2d(32)
         ComplexModReLU(32)
         Output complex shape: [B, 32, ceil(E/2), ceil(F/4)]

Block 3: ComplexConv2d 32 -> 64, kernel 3x3, stride (2, 2), padding 1
         ComplexBatchNorm2d(64)
         ComplexModReLU(64)
         Output complex shape: [B, 64, ceil(E/4), ceil(F/8)]

Block 4: ComplexConv2d 64 -> 64, kernel 3x3, stride (1, 2), padding 1
         ComplexBatchNorm2d(64)
         ComplexModReLU(64)
         Output complex shape: [B, 64, ceil(E/4), ceil(F/16)]
```

After the complex feature extractor:

```text
1. Split output into real and imaginary maps.
2. Convert to magnitude: sqrt(real^2 + imag^2 + 1e-8).
3. AdaptiveAvgPool2d((1, 1)).
4. Flatten.
5. Linear(64, 64).
6. ReLU.
7. Dropout(p=0.15 by default).
8. Linear(64, 1).
```

The final scalar is the normalized range prediction. During training, labels are
normalized as:

```text
y_norm = (y_range_km - train_mean_km) / train_std_km
```

During reporting and prediction, the model output is converted back to km:

```text
pred_range_km = pred_norm * train_std_km + train_mean_km
```
