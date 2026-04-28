# Prediction Workflow

## Basic Prediction

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

## Outputs

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
