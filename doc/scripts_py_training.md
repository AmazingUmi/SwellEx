# Training Workflow

## Basic Training

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

## Model Selection

Use `--model complex_cnn_range` to select the complex model:

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
files, pass both `--train-data` and `--val-data`.

## Outputs

Training writes:

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

## Resume Training

When a previous `<dataset>_last_MMDD_HHMMSS.pt` exists under
`outputs/networks_results/<model_name>/train_outputs`, `train` automatically
continues from the newest checkpoint for the selected `--model` and `--data`.
In this case `--epochs` is the number of additional epochs to run.

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
