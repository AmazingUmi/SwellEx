# Neural Training and Prediction

Direct **base-pack** Python CLI reference (automated runs use `run.sh` in [remote validation](../remote_workflow/run-and-validate.md)).

## Manual Train and Predict

[Enter a frozen run](../remote_workflow/remote-environment.md#manual-commands-in-a-frozen-run) on `haiqin` and select a dataset from `report.json`:

```bash
NEURAL_OUTPUT='outputs/manual_results/rbd-complex-cnn'
"$SWELLEX_PYTHON" -B scripts/network_training/neural/main.py train \
  --method RBD --model complex_cnn_range --data '<rbd_dataset_name>' --device cuda \
  --output-dir "$NEURAL_OUTPUT" --no-resume --epochs 100 --loss-space km --huber-beta 0.5

"$SWELLEX_PYTHON" -B scripts/network_training/neural/main.py predict \
  --method RBD --model complex_cnn_range --data '<rbd_dataset_name>' --device cuda \
  --checkpoint '<path_to_best_checkpoint>' --output-dir "$NEURAL_OUTPUT"
```

One entry, `scripts/network_training/neural/main.py {train,predict}`, with a required `--method`; `--model` must belong to it:

| `--method` | `--model` choices (registry names in `scripts/network_training/catalog.py`) | Default `--output-dir` |
|---|---|---|
| `RBD` | `complex_cnn_range`, `resnet18_range`, `resnet50_range` | `outputs/networks_results/RBD` |
| `SCM` | `scm_complex_cnn_range`, `scm_resnet18_range`, `scm_resnet50_range` | `outputs/networks_results/SCM` |

Shared CLI options: `--method`, `--data DATASET` (under `outputs/Datasets/` or path; `predict` reads `*_test.h5`), `--model MODEL`, `--device` (default CUDA if available), `--output-dir`, `--batch-size` (CLI default `16` train / `32` predict; workflow TOML `64`), and `--num-workers` (`0`).

## Train Options

| Option | Default | Meaning |
|---|---|---|
| `--train-data GLOB ...` / `--val-data GLOB ...` | `*_train.h5` / auto | Validation order: `--val-data` → `*_val.h5` → seeded `--val-fraction` (`0.25`) split |
| `--epochs N` / `--seed N` | `100` / `2026` | Fresh or additional resume epochs; split seed (`0..2**32-1`) |
| `--lr` / `--weight-decay` | `3e-4` / `1e-4` | AdamW hyperparameters |
| `--base-channels` / `--dropout` | `16` / `0.15` | Fresh CNN width; fresh dropout in `[0, 1)` |
| `--loss-space` / `--huber-beta` | `normalized` / `0.5` | `normalized` or `km` SmoothL1 (workflow TOML uses `km`); MAE/RMSE always reported in km |
| `--no-input-norm` | Off | Disable per-sample real/imag normalization |
| `--no-resume` / `--resume-checkpoint PATH` | Off / auto | Start fresh or resume a specific checkpoint (mutually exclusive) |

## Predict Options, Resume, and Checkpoint Lookup

`predict` accepts `--checkpoint PATH`, `--predictions-csv PATH`, `--plot-path PATH` (`.pdf`), `--no-plot`, and `--print-first N` (`10`).

- **Automatic lookup**: `train` (without `--no-resume`) checks `<dataset>_last.pt` then latest-mtime `MMDD_HHMMSS` `last` file via `network_training/common/paths.py:latest_time_suffixed_path` (starting fresh if none exists); `predict` checks `<dataset>_best.pt` then latest-mtime `best` file. Explicit missing paths fail.
- **Resume contract**: Restores weights, config, target statistics, loss settings, optimizer state, and history after verifying `input_signature`, split mode/indices, and seed. Each `train` call starts a fresh cosine LR decay from `--lr` to `0` over `--epochs`.

## Remote Artifacts and Local Delivery

Under `outputs/networks_results/<METHOD>/<model>/`: `train_outputs/<dataset>_{best,last,history}_MMDD_HHMMSS.{pt,json}` and `test_outputs/<dataset>_predictions_MMDD_HHMMSS.csv` (`sample,source_segment_idx,window_center_s,pred_range_km,true_range_km,abs_error_km`). Automated `fetch` delivers CSVs/histories/logs to `output/data/<run_id>/` and PDFs to `output/figure/<run_id>/`.
