# MoE Data Scale Pack

`scripts/MoE_data_scale/` varies the number of independent residual experts
while keeping their structure and the shared frontend fixed. It supports RBD,
ELM, SCM, and the RBD+SCM combination. The expert-count rule is uncalibrated;
the fixed-expert matrix measures capacity without applying that rule.

## RBD+SCM Data-Size Matrix

```bash
# Inspect all 48 cells locally; no SSH, files, or training are started.
./scripts/run.sh MoE_data_scale study --plan scripts/MoE_data_scale/localization_networks/workflow/rbd_scm_expert_study_v1.json --dry-run

# Launch/recover the complete matrix on haiqin.
./scripts/run.sh MoE_data_scale study --plan scripts/MoE_data_scale/localization_networks/workflow/rbd_scm_expert_study_v1.json --study-dir output/data/rbd-scm-moe-expert-study-v1
./scripts/run.sh MoE_data_scale status --run 'rbd-scm-moe-expert-study-v1'
./scripts/run.sh MoE_data_scale fetch --run 'rbd-scm-moe-expert-study-v1'

# A single joint-input control.
./scripts/run.sh MoE_data_scale --preset train_before --method RBD_SCM --network complex_MoE_e4
```

The plan fixes S5 `train_before`, 100 epochs, batch size 64, base channels 16,
balance-loss weight 0.01 and seeds 2026/2027/2028. Every cell trains one fixed
expert count; auto selection is excluded. Each independent expert retains
the architecture used in the previous SCM-only experiment.

| Fraction of optimizer-training candidates | Expected training windows | Expert counts | Repeats |
|---:|---:|---|---:|
| 0.125 | 75 | 1, 2, 4, 8 | 3 seeds |
| 0.25 | 150 | 1, 2, 4, 8 | 3 seeds |
| 0.5 | 300 | 1, 2, 4, 8 | 3 seeds |
| 1.0 | 600 | 1, 2, 4, 8 | 3 seeds |

This is 16 conditions and 48 training runs. Counts are expectations for the
current S5 configuration; artifacts record actual retained valid unique rows.
Within each seed, the 200 validation windows stay fixed and training subsets
are nested. All cells use the same 700 held-out test windows. Different seeds
also change the validation split. Fixed 100 epochs give 200/300/500/1000 optimizer
updates across these data sizes; this is not an equal-update-budget experiment.

`RBD_SCM` is the canonical name and concatenation order: 21 RBD rows followed
by 231 SCM rows, normally `[batch,2,252,22]`. The combined axis is feature rows.
The pack checks sample identities, labels, window geometry, physical axes and
source provenance before concatenation. It fixes `block_rms`: one complex RMS
scale per block fitted **after** training subsampling, using only that cell's
retained optimizer-training rows. Validation, prediction and resume reuse the
saved state. No test-fitted scales or additional fusion modes are selected.
At a given seed/fraction, all expert counts use identical training rows and
preprocessing. RMS equalizes per-row power, not total block energy.

The detached remote coordinator freezes the plan/source, serializes child runs,
and survives local disconnection. Re-entering the same study follows existing
children; failed or ambiguous attempts are not silently retrained. Changing
parameters requires a new study directory. Fetch places child reports/figures
under `output/{data,figure}/<study_id>/children/<run_id>/` and summarizes available
cells with missing results explicitly marked. Complete groups report means and
sample standard deviations over seeds; expert selection uses validation RMSE,
not test error. These same-record results do not establish unknown-environment
generalization or a calibrated data-to-expert rule.

The study directory contains `study-cells.csv` (individual runs),
`study-summary.csv` (complete seed groups), `validation-selection.csv`
(validation-selected expert count per data size), and `study-report.json`
(identity checks and missing-cell status). Summaries include MAE/RMSE, training
wall time, parameter counts, optimizer steps, and best-epoch routing statistics.

## Architecture Check and Recovery

```bash
# Synthetic CUDA checks only: no raw feature generation or experiment training.
./scripts/run.sh MoE_data_scale validate --suite quick
./scripts/run.sh MoE_data_scale status --run 'RUN_ID'
./scripts/run.sh MoE_data_scale fetch --run 'RUN_ID'
./scripts/run.sh MoE_data_scale --help
```

Use the complete ID printed after `Run:`. Reports and logs are delivered to
`output/data/<run_id>/`. The quick report records synthetic inputs, devices,
actual tensor shapes, parameter counts, and the scope of each check. A quick
PASS is not evidence of regression accuracy, expert specialization, or a useful
data-to-expert scaling rule. Numerical checks run on haiqin from a frozen source
snapshot with the existing frozen uv environment and explicit CUDA.

Pack-only invocation follows the shared contract and starts its full configured
experiment. It is **not** the architecture-only command above. `study` runs the
explicit matrix; `budget-study` is not provided by this pack.

## Model and Direct Connections

The shared complex frontend has channels `c, 2c` and strides `(1,2), (2,2)`.
Each expert independently owns a `2c -> 4c` projection with stride `(2,1)`,
followed by an equal-width residual block:

```text
projected feature x -> complex conv -> normalization -> modReLU
                    -> complex conv -> normalization -> F(x)
expert residual output = x + F(x)
```

The shortcut is an exact identity: residual convolutions have stride one and
there is no activation after addition. A zero residual branch must therefore
return `x` unchanged. Magnitude pooling and a fixed `4c -> 4c -> 1` real head
produce a normalized range prediction. Expert normalization uses separate
real/imaginary GroupNorm; it does not use routed mini-batch statistics.
Expert structures match, but their learned parameters are independent.

The router pools shared-feature magnitudes and applies `Linear(2c, E)`.
Increasing E changes only its output dimension, not an additional hidden width
or depth. Per-sample Top-2 routing evaluates only selected experts, then mixes
their predictions using normalized selected weights. Samples are never dropped.
E=1 bypasses routing; E=2 evaluates both experts and has no sparse-compute gain.
Parameters and optimizer memory grow with E; fixed Top-2 does not guarantee
constant wall-clock cost on one GPU.

| Public network | Expert selection |
|---|---|
| `complex_MoE` | Resolve from retained training count before initialization |
| `complex_MoE_e1` | One expert; within-family baseline |
| `complex_MoE_e2` | Two experts |
| `complex_MoE_e4` | Four experts |
| `complex_MoE_e8` | Eight experts |

The full plan contains all five networks for each of RBD/ELM/SCM/RBD_SCM, 20 models.
Smoke uses fixed E1 and E4 for each input, eight models. The public shell rejects
GRNN, `--fusion` overrides, old CNN depth names, and networks from other packs.
RBD_SCM's fixed fusion policy is recorded in its dataset/checkpoint signature
and fitted preprocessing metadata; existing single-route MoE run IDs are unchanged.

## Count Rule and Training Contract

N is the valid unique optimizer-training window count after validation splitting
and nested training subsampling. Epochs, repeated batches and feature dimensions
do not increase N. Data selection preserves the existing physical-axis and
source-membership checks. `adapt` remains rejected because its current fitting
stage precedes the retained optimizer subset.

```toml
[scaling]
reference_samples = 600
reference_experts = 2
exponent = 0.5
max_experts = 8

[moe]
balance_loss_weight = 0.01
```

Auto computes `reference_experts * (N / reference_samples) ** exponent`, clips
to the supported range and rounds down to an allowed expert count in
`1,2,4,8`. These are starting hypotheses, not calibrated thresholds. Fixed
network controls keep their declared expert count. The resolved model stores
the expert count, Top-k, fixed layout, router/normalization contracts and exact
total and per-sample active parameter counts before optimization. Prediction
rebuilds the saved model without selecting experts from prediction data size.

The training objective combines SmoothL1 regression and the weighted routing
balance loss. Histories record regression loss, total objective, balance loss,
router entropy, expert assignment counts and mean selected weights separately.
Validation RMSE alone selects the best checkpoint. Checkpoints require pack,
method, physical axes, complete architecture and balance-loss-weight identity;
incompatible or incomplete artifacts are rejected. The model's public
`forward` still returns normalized `[batch,1]`; training obtains auxiliary
values through a single `forward_with_aux` evaluation.

No new dependencies are required. Generation, prediction, and visualization
retain the shared [remote workflow](../workflows/remote-validation.md) and
[dataset contracts](../datasets/hdf5-datasets-and-splits.md).

## Recorded Architecture Validation

On 2026-10-16 at 21:24 Singapore time,
`20261016T132432Z-moe-full-all-all-b0e9029b` completed **PASS** on haiqin's
RTX A5000 with PyTorch `2.11.0+cu128`. The quick report contains 26 PASS checks,
including all 15 registrations at the default `c=16`, a full
`[64,2,231,23]` SCM E4 forward/backward, and seven focused residual, sparse
dispatch, singleton, gradient, optimizer and checkpoint contracts. Its nested
workflow/artifact suite contains 19 PASS tests. Local standard-library routing
and artifact regressions additionally passed all 38 tests.

The measured trainable counts match the frozen formulas:

| Experts | Total parameters | Per-sample active parameters |
|---:|---:|---:|
| 1 | 199,185 | 199,185 |
| 2 | 388,692 | 388,692 |
| 4 | 767,640 | 388,758 |
| 8 | 1,525,536 | 388,890 |

Active counts include the shared frontend, selected experts and the complete
router. A batch may execute more than two distinct experts across its samples.
The identity check verifies both exact `x + 0 = x` output and unit shortcut
gradient. Synthetic optimizer checks exercise six updates, without using
SwellEx observations or measuring scientific performance.

Evidence is in
`output/data/20261016T132432Z-moe-full-all-all-b0e9029b/report.json` and
`reports/python_quick.json`. The frozen source SHA256 is
`bce4624b839c3af4859a7dd22001d8288e0e584b0870f9e49ca08c03061e3b4a`.
The worker verified snapshot integrity before and after validation.
That architecture-only run did not perform real-data training or a data-size study.

## Recorded Single-Model Run

`20261016T140024Z-moe-before-SCM-complex_MoE_e4-3c362a7f` subsequently completed
the full SCM-only workflow on haiqin: 600 training, 200 validation and 700 test
windows; 100 epochs; seed 2026; best epoch 86. Test MAE/RMSE were
104.53/145.68 m, best validation RMSE 19.63 m, and training wall time 70.75 s.
All ten workflow phases and nine delivery files passed. This is one fixed E4
result, not the RBD+SCM matrix or evidence of an expert-count scaling benefit.

## Recorded Matrix Workflow Validation

On 2026-10-16, `20261016T145605Z-moe-full-all-all-2a00f8f6` completed **PASS**
with 38 quick checks on haiqin, including all 20 model variants, the RBD+SCM
default batch shape, raw-generation acceptance/rejection controls, and joint
alignment, subset-only RMS fitting, checkpoint restoration and sampler isolation.
Its nested standard-library suite passed 61 tests; the broader local dispatch,
study and artifact regression suite passed 80 tests.

`rbd-scm-moe-matrix-smoke-20261016-v2` completed **4/4 children PASS** using
75/600 training windows × E1/E4, seed 2026, and one epoch per child. All four
retained 200 validation and 700 test windows. Fetch and aggregation verified
identical joint HDF5 bytes, fixed validation/test identities, nested training
subsets, and identical fitted scales across expert counts at each data size.
The reports, three summary tables and child figures were delivered under
`output/{data,figure}/rbd-scm-moe-matrix-smoke-20261016-v2/`.

The first smoke attempt (`...-v1`) stopped before training because the raw
generator validator incorrectly required the combined method; the raw and
joint validation scopes were separated before the successful fresh run.
The one-epoch smoke verifies engineering integration only. The bundled plan's
dry-run expands to 48 unique children without launching training.

## Recorded Formal Matrix

`rbd-scm-moe-expert-study-v1` completed **48/48 children PASS** on 2026-10-16
at 23:43 Singapore time. The retrieved study report was verified on 2026-10-17:
all 16 conditions contain three seeds, each child completed 100 epochs, and
there are no missing cells or identity-validation errors. Actual optimizer
training counts were 75/150/300/600; every child retained 200 validation and
700 test windows under the split and preprocessing contracts above.

Test MAE in metres (mean ± sample SD over three seeds):

| Training windows | E1 | E2 | E4 | E8 |
|---:|---:|---:|---:|---:|
| 75 | 360.28 ± 137.74 | 359.99 ± 142.42 | 343.12 ± 133.41 | 371.62 ± 125.43 |
| 150 | 175.84 ± 24.15 | 171.57 ± 25.83 | 177.24 ± 24.69 | 187.92 ± 29.09 |
| 300 | 131.96 ± 13.62 | 130.27 ± 16.40 | 129.25 ± 12.63 | 144.26 ± 23.71 |
| 600 | 115.92 ± 4.90 | 116.64 ± 5.00 | 117.71 ± 10.35 | 128.23 ± 8.90 |

Across-seed mean best validation RMSE selected E4/E4/E8/E4 respectively.
At 600 windows, E4's test RMSE was 165.10 ± 13.74 m, versus E1's
169.94 ± 1.91 m; E1 had the lower mean test MAE. Increasing expert count did
not consistently improve MAE, and these results do not calibrate a monotonic
data-to-expert rule. Fixed epochs also increase optimizer updates with data
size, so this matrix does not isolate data quantity from optimization budget.

Training wall time summed to 35.57 minutes; the first-child start to last-child
finish spanned 42.64 minutes including generation, prediction and plotting.
At 600 windows, mean training time per run was 59.55/71.10/73.86/88.90 s for
E1/E2/E4/E8. These are recorded workflow timings, not controlled throughput
benchmarks. Evidence and complete MAE/RMSE/routing/cost summaries are under
`output/data/rbd-scm-moe-expert-study-v1/`; child PDFs are under the matching
`output/figure/` study directory. Generated artifacts remain outside Git.
