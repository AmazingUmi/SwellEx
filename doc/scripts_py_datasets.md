# HDF5 Datasets and MATLAB Split Strategies

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
  both sides, and use one side for training and the other for testing.

Configure `Range_nearby` with:

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

## Dataset Contract

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

## Random Split From Training Files

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

## Fixed Train/Validation Split

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

## Override Training Files

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

## Predict On A Different Test Set

Prediction data only needs to match the model input shape and HDF5 field names:

```powershell
python scripts_py/Network_main.py predict `
  --model complex_cnn_range `
  --data periodic_4_1
```

The default fixed checkpoint path is a convenience alias. If it does not exist,
the newest matching `best_*.pt` file in the model's `train_outputs` directory is
used automatically.

## Add A New Dataset Format

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
