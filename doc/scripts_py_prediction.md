# Prediction Workflow

Prediction also uses method-specific entry points.
Shared CSV writing and plotting helpers live in `scripts_py/common/`, while
method-specific prediction modules still choose the dataset loader and model
registry.

## RBD Prediction

```bash
python3 scripts_py/RBD_method/Network_main.py predict \
  --model complex_cnn_range \
  --data Range_nearby_after_800s_gap15s_test_no_beamformer
```

## ELM Prediction

```bash
python3 scripts_py/ELM_method/Network_main.py predict \
  --model elm_complex_cnn_range \
  --data periodic_4_1_elm_pairwise_ratio_upper_mel64
```

If `--checkpoint` is omitted, prediction first tries:

```text
outputs/networks_results/<method>/<model>/train_outputs/<dataset>_best.pt
```

If that fixed path does not exist, it uses the newest timestamped:

```text
<dataset>_best_MMDD_HHMMSS.pt
```

## Outputs

RBD predictions:

```text
outputs/networks_results/RBD_method/<model>/test_outputs/<dataset>_predictions_MMDD_HHMMSS.csv
outputs/networks_results/RBD_method/<model>/test_outputs/<dataset>_range_prediction_MMDD_HHMMSS.png
```

ELM predictions:

```text
outputs/networks_results/ELM_method/<model>/test_outputs/<dataset>_predictions_MMDD_HHMMSS.csv
outputs/networks_results/ELM_method/<model>/test_outputs/<dataset>_range_prediction_MMDD_HHMMSS.png
```

Prediction CSV fields:

```text
sample
source_segment_idx
window_center_s
pred_range_km
true_range_km
abs_error_km
```

Use `--no-plot` to skip plot generation, or `--plot-path` to choose another
plot path.
