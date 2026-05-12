# Training Workflow

The Python training code is now split by method.

```text
scripts_py/common/
scripts_py/RBD_method/Network_main.py
scripts_py/ELM_method/Network_main.py
```

`scripts_py/common/` contains shared path helpers, HDF5 path/split helpers,
training utilities, and prediction export/plot utilities. RBD and ELM still keep
their own dataset loaders and model registries because their HDF5 layouts differ.

Both commands read datasets from `outputs/Datasets/<dataset_name>/` and write
results under method-specific output roots:

```text
outputs/networks_results/RBD_method/
outputs/networks_results/ELM_method/
```

## RBD Training

RBD HDF5 input:

```text
/X: [sample, element, frequency, real_imag]
torch: [batch, 2, element, frequency]
```

Example:

```bash
python3 scripts_py/RBD_method/Network_main.py train \
  --model complex_cnn_range \
  --data Range_nearby_after_800s_gap15s_test_no_beamformer
```

Physical-error loss example:

```bash
python3 scripts_py/RBD_method/Network_main.py train \
  --model complex_cnn_range \
  --data Range_nearby_after_800s_gap15s_test_no_beamformer \
  --loss-space km --huber-beta 0.5
```

Available RBD models:

- `complex_cnn_range`
- `real_cnn_range`
- `resnet18_range`
- `resnet50_range`

## ELM Training

ELM HDF5 input:

```text
/X: [sample, pair, frequency, real_imag]
torch: [batch, 2, pair, frequency]
```

where:

```text
pair contains strict upper-triangle element pairs with i < j
pair_count = element_count * (element_count - 1) / 2
```

Example:

```bash
python3 scripts_py/ELM_method/Network_main.py train \
  --model elm_complex_cnn_range \
  --data periodic_4_1_elm_pairwise_ratio_upper_mel64
```

Physical-error loss example:

```bash
python3 scripts_py/ELM_method/Network_main.py train \
  --model elm_complex_cnn_range \
  --data periodic_4_1_elm_pairwise_ratio_upper_mel64 \
  --loss-space km --huber-beta 0.5
```

Available ELM models:

- `elm_complex_cnn_range`
- `elm_real_cnn_range`
- `elm_resnet18_range`
- `elm_resnet50_range`

## Common Options

```bash
--epochs 100
--batch-size 16
--lr 3e-4
--weight-decay 1e-4
--dropout 0.15
--huber-beta 0.5
--loss-space normalized
--val-fraction 0.25
--no-resume
--resume-checkpoint <path>
--train-data "outputs/Datasets/<dataset>/*_train.h5"
--val-data "outputs/Datasets/<dataset>/*_val.h5"
```

`--loss-space normalized` keeps the original target-normalized SmoothL1 loss.
`--loss-space km` computes SmoothL1 directly on physical range error in
kilometers. With `--loss-space km`, `--huber-beta` is interpreted in kilometers.

If `*_val.h5` is not present and `--val-data` is omitted, the selected training
files are split into train/validation sets by `--seed` and `--val-fraction`.

## Outputs

RBD example:

```text
outputs/networks_results/RBD_method/<model>/train_outputs/<dataset>_best_MMDD_HHMMSS.pt
outputs/networks_results/RBD_method/<model>/train_outputs/<dataset>_last_MMDD_HHMMSS.pt
outputs/networks_results/RBD_method/<model>/train_outputs/<dataset>_history_MMDD_HHMMSS.json
```

ELM example:

```text
outputs/networks_results/ELM_method/<model>/train_outputs/<dataset>_best_MMDD_HHMMSS.pt
outputs/networks_results/ELM_method/<model>/train_outputs/<dataset>_last_MMDD_HHMMSS.pt
outputs/networks_results/ELM_method/<model>/train_outputs/<dataset>_history_MMDD_HHMMSS.json
```

Checkpoints store:

- `model_name`
- `model_config`
- model weights
- optimizer and scheduler state
- target normalization statistics
- loss space and Huber beta
- input normalization flag
- train/validation split metadata
- source HDF5 paths
- full training history
