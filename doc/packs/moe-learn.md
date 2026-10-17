# Continual MoE Demo Pack

```bash
# Leave out the complete near, middle or far distance interval, respectively.
./scripts/run.sh MoE_learn --config scripts/MoE_learn/localization_networks/workflow/haiqin_holdout_near.toml
./scripts/run.sh MoE_learn --config scripts/MoE_learn/localization_networks/workflow/haiqin_holdout_middle.toml
./scripts/run.sh MoE_learn --config scripts/MoE_learn/localization_networks/workflow/haiqin_holdout_far.toml
# Append --dry-run to inspect any configuration without launching numerical work.

# Earlier protocol: group after stages by distance, with test rows in all groups.
./scripts/run.sh MoE_learn --config scripts/MoE_learn/localization_networks/workflow/haiqin_before_then_after_range.toml --dry-run
./scripts/run.sh MoE_learn --config scripts/MoE_learn/localization_networks/workflow/haiqin_before_then_after_range.toml

# Compare with the original three after batches interleaved by window time.
./scripts/run.sh MoE_learn --config scripts/MoE_learn/localization_networks/workflow/haiqin_before_then_after.toml --dry-run
./scripts/run.sh MoE_learn --config scripts/MoE_learn/localization_networks/workflow/haiqin_before_then_after.toml

# Inspect the resolved demo locally; no SSH or numerical computation.
./scripts/run.sh MoE_learn --dry-run

# Run the small real-data demo on haiqin and follow its progress.
./scripts/run.sh MoE_learn
# Equivalent explicit selection:
./scripts/run.sh MoE_learn --preset demo --method RBD_SCM --network complex_MoE_learn

# Synthetic CUDA contracts, then a short real-data integration check.
./scripts/run.sh MoE_learn validate --suite quick
./scripts/run.sh MoE_learn validate --suite smoke

# Recover using the complete ID printed after Run:.
./scripts/run.sh MoE_learn status --run 'RUN_ID'
./scripts/run.sh MoE_learn fetch --run 'RUN_ID'
./scripts/run.sh MoE_learn --help
```

Reports, per-stage prediction CSVs and deployment checkpoints are fetched to
`output/data/<run_id>/`; stage-learning figures are under
`output/figure/<run_id>/`. Numerical work uses an isolated source snapshot and
the frozen uv environment on haiqin, requires CUDA, and survives local
disconnection. See the [remote workflow](../workflows/remote-validation.md).

This pack studies sequential data arrival, adaptation of existing experts at a
smaller learning rate, router adaptation and candidate expert growth. The
default `demo` is a short engineering check. Distance-holdout configurations
exclude an entire distance interval from learning and evaluate three deployments.
Earlier before-to-after configurations evaluate four deployments with test rows
drawn from the same distance intervals as training. These protocols are not
data-size matrices or evidence that
larger datasets always need more experts. Controller thresholds are uncalibrated.

## Commands and Configuration

`MoE_learn` owns its code and artifacts under `scripts/MoE_learn/`. Its sole
method is `RBD_SCM`, its network is `complex_MoE_learn`, and its only bundled
preset is `demo`. Fusion is fixed to `block_rms`. Foreign methods/networks,
fusion overrides, legacy aliases and unknown configuration fields are rejected.

Pack-only invocation runs the configured real-data demo. `validate` defaults
to `smoke`; `quick` uses synthetic inputs and does not require `Origindata`.
Both experiment and validation commands accept `--dry-run`. A complete custom
TOML is selected with `--config PATH`, mutually exclusive with `--preset`.
Status/fetch accept the same custom `--config` when a different remote location
was used. Runs use fresh feature generation; no old pack's dataset or checkpoint
is silently reused.

The bundled source of truth is
[`haiqin_demo.toml`](../../scripts/MoE_learn/localization_networks/workflow/haiqin_demo.toml).
The distance-holdout protocols use
[`haiqin_holdout_near.toml`](../../scripts/MoE_learn/localization_networks/workflow/haiqin_holdout_near.toml),
[`haiqin_holdout_middle.toml`](../../scripts/MoE_learn/localization_networks/workflow/haiqin_holdout_middle.toml)
and [`haiqin_holdout_far.toml`](../../scripts/MoE_learn/localization_networks/workflow/haiqin_holdout_far.toml).
The earlier protocols use complete custom configurations:
[`haiqin_before_then_after_range.toml`](../../scripts/MoE_learn/localization_networks/workflow/haiqin_before_then_after_range.toml)
groups after rows by distance;
[`haiqin_before_then_after.toml`](../../scripts/MoE_learn/localization_networks/workflow/haiqin_before_then_after.toml)
keeps the original time-interleaved groups. Select these files with `--config`;
they are not additional `--preset` names.

| Setting | Demo default | Meaning |
|---|---|---|
| Dataset interval | S5, 0–119 s | Small same-record input interval |
| Input | RBD then SCM | Aligned feature rows with fixed `block_rms` |
| Dataset split | `full_interleaved`, 4:1 | Separate training-source and fixed test rows |
| Stage mode / count | `interleaved` / 2 | Disjoint groups of training-source rows |
| Per-stage validation | 0.2 | Held out before fitting or optimization |
| Initial / adaptation / warmup epochs | 3 / 2 / 1 | Separate small training budgets |
| Batch size / base channels | 8 / 16 | Fixed frontend and per-expert structure |
| Initial / maximum experts | 2 / 4 | Expansion adds one expert per accepted stage |
| Replay mode / capacity | `buffer` / 16 | Reservoir of prior optimizer-training rows |
| Historical validation | `true` | Evaluate candidate regressions on earlier validation rows |
| Minimum relative improvement | 0.01 | Current validation RMSE improvement required for acceptance |
| Maximum old relative regression | 0.05 | Historical validation guard when available |
| Novelty threshold | 3.0 | Trigger for an expansion trial, not an acceptance guarantee |
| Balance-loss weight | 0.01 | Auxiliary router objective, separate from regression metrics |

Training uses separate learning-rate settings:

| `training` field | Default | Parameters updated |
|---|---:|---|
| `learning_rate` | `1e-3` | All model parameters during initial training |
| `old_expert_learning_rate` | `1e-4` | All experts present at the start of a later stage |
| `new_expert_learning_rate` | `1e-3` | The expert added during the current stage, including direct warmup |
| `router_learning_rate` | `1e-3` | Router during later-stage adaptation |

All four rates must be finite and positive, and
`old_expert_learning_rate < new_expert_learning_rate`. The initial
`learning_rate` does not override the later parameter groups. The supplied
holdout and before-to-after configurations use these same rates.

Smoke limits initial/adaptation/warmup training to at most one epoch. For a
`full_interleaved` configuration it also caps the interval at its start plus
39 seconds. Submitted and resolved configurations are stored separately.
Inspect custom smoke settings with `validate --suite smoke --dry-run`.

## Complete Distance Holdout

`dataset.stage_mode="distance_holdout"` runs three separate folds with
`distance_edges_km=[0.90, 1.32, 1.74, 2.16]` and `heldout_group=1`, `2` or `3`:

| Fold | Complete test distance interval (km) | After training increments, in order |
|---|---|---|
| Near | `[0.90, 1.32)` | Middle, far |
| Middle | `[1.32, 1.74)` | Near, far |
| Far | `[1.74, 2.16]` | Near, middle |

For each fold, exclude every before-side row in its held-out interval from
training and validation. Every after-side row in that interval becomes fixed
test data. None of these before exclusions or after test rows may enter
preprocessing fitting, optimizer updates, validation-based selection or replay.
The remaining before rows split 80:20 for stage-one training and validation.
The remaining after rows first split 80:20 overall, then their training and
validation subsets are each assigned to the other two distance intervals,
introduced near to far in stages two and three. Counts differ across folds;
the earlier 640/160 and 448/112/140 allocation does not apply here.

Each fold fits block RMS, target statistics and the descriptor reference only
from its own stage-one training rows. Its test membership stays fixed through
all three stages. The supplied folds retain 30 epochs for each initial,
adaptation and warmup fit, the documented separate learning rates, and full
replay of previously permitted training rows. Warmup still uses current-stage
training only. Test metrics report deployed models and never select them.

The three test subsets are disjoint and together cover the same 700-row after
pool. They are different evaluation subsets, not repeated measurements of one
test population; their stage curves have no confidence intervals. The middle
fold examines a missing interior distance interval, while the near and far
folds examine withheld distance extremes. All use known trajectory labels to
define the split and remain a same-record evaluation, not an unseen sea area
or online discovery without labels.

## Earlier Before-to-After Protocols

`before_then_after` with `range_bins` groups newly introduced training data by
distance, but still trains and validates at every test distance interval. It
does **not** implement complete distance holdout; use `distance_holdout` above
for that question.

`dataset.stage_mode="before_then_after"` requires
`split_strategy="range_overlap"` and `train_side="before"`.
`after_test_fraction` and `after_grouping` are required only in this mode and rejected in the other
modes. Both supplied configurations use `after_test_fraction=0.2`,
`val_fraction=0.2`, four stages, full replay, and seed 2026.

The complete before pool is split once into stage-one training and validation.
The after pool is allocated **before training**: first reserve its fixed test
fraction, then split the remainder into optimizer-training and validation pools.
Both grouping modes preserve exactly the same before and after
training/validation/test identities. Only the assignment of after training and
validation rows to stages changes. The fixed after test set is never among
these increments or replay inputs.

With `after_grouping="time_interleaved"`, sort the after training and validation
pools separately by window time, then distribute each across three interleaved
increments. Expected counts for this mode are below; generated metadata and
stage reports record actual retained rows.

| Deployment | Newly available training rows | Validation rows introduced | Cumulative after training rows | Fixed after test rows |
|---|---:|---:|---:|---:|
| Stage 1: before only | 640 before | 160 before | 0 | 140 |
| Stage 2: first after batch | 150 after | 38 after | 150 | 140 |
| Stage 3: second after batch | 149 after | 37 after | 299 | 140 |
| Stage 4: third after batch | 149 after | 37 after | 448 | 140 |

These counts start from 800 before and 700 after windows. Reserving 140 after
test windows leaves 560; the latter contribute 448 training and 112 validation
windows. These totals also apply to the distance-grouped configuration.

With `after_grouping="range_bins"`, use
`after_range_edges_km=[0.90, 1.32, 1.74, 2.16]` to assign the same after training
and validation rows to three groups, introduced from near to far:

| Deployment | Newly introduced after distance interval (km) |
|---|---|
| Stage 1 | None; before-only training and validation |
| Stage 2 | `[0.90, 1.32)` |
| Stage 3 | `[1.32, 1.74)` |
| Stage 4 | `[1.74, 2.16]` |

The first two intervals exclude their right endpoint; the last includes it.
Group sizes need not be equal; do not reuse the time-interleaved per-stage
counts for this mode. The fixed 140-row test set retains all three distance
groups. After each deployment, report both overall test errors and errors
within each fixed distance group.
Edges must be finite, positive, strictly increasing and have length equal to
`continual.stages`. They must cover the complete after pool; every group must
contain training, validation and test rows. Invalid or empty groups are rejected.

Source IDs, raw sample support intervals and physical time windows
must be disjoint across training/validation/test roles. The complete before
pool must precede the after pool. Assignment seeds and membership lists are
saved. Distance grouping uses trajectory ground truth; it is a prescribed
label-based sequence, not online discovery without labels or chronological
arrival. Non-overlapping distance intervals apply only to newly introduced
rows across after stages. Before training can cover those same distances,
and full replay retains earlier training rows during subsequent adaptation.

Both supplied runs use 30 initial epochs, 30 existing-expert/router adaptation
epochs and 30 new-expert warmup epochs. When an expansion is trialled, its
joint old-expert, new-expert and router fit gets another 30 adaptation epochs,
with each parameter group using its configured rate. Both candidate trials
are additional work, so these are per-fit budgets, not 30 total epochs per
stage or an equal-compute comparison. Accepted deployment state, including
an unchanged `REUSE` decision, is saved and tested after every stage.

## Data and Adaptation Boundaries

The generator produces aligned RBD and SCM blocks with matching labels,
physical frequency/array/pair axes, source identities and time windows. RBD
rows precede SCM rows; the combined axis is feature rows. The loader rejects
foreign-pack data, mismatched physical axes or source provenance, invalid
samples, and training/test source overlap.

For `interleaved` and `chronological`, training-source rows are sorted by
window time before assigning stages.
`interleaved` takes every kth row for each of k stages; it is deterministic
same-record interleaving, not an independent random-domain stream.
`chronological` assigns consecutive time blocks. Each stage is then split into
optimizer-training and validation rows with the recorded seed. These groups
are disjoint and cover the training-source rows exactly once.

Only stage one's optimizer-training rows fit the two block RMS scales, target
mean/standard deviation and descriptor reference. Later stages, historical
validation and test rows reuse those fitted values. Per-sample real/imaginary
standardization remains a deterministic input transform. Stage memberships,
fit counts and source-identity hashes are persisted.

This pack's `moe-learn-python-features-v3-window-centered` generator subtracts
each element's mean within its own window. Other windows do not determine
that centering; both `before_then_after` and `distance_holdout` reject older
generated features.
Range-overlap pool construction still uses both sides' trajectory labels to
define their shared range. Thus this protocol evaluates adaptation within
a known shared range, with correlated neighbouring windows from one recording.
Disjoint support does not make the observations independent or demonstrate
generalization to a new sea area, voyage or recording environment.

After initial training, the shared frontend stays frozen and in evaluation
mode, keeping the descriptor coordinates fixed. Its parameter digest must
remain unchanged during candidate training. Block RMS scales and target
statistics also remain fixed. Existing expert branches continue training at
the smaller old-expert rate; their parameters are not permanently frozen.
An accepted new expert joins the old-expert group at the next stage.

Deployment checkpoints store an evaluation model with expert gradients
disabled for inference. A later-stage candidate explicitly enables the
appropriate expert/router gradients and rebuilds the optimizer groups.
This inference state is not a lifetime freeze policy. Neither a smaller
expert rate nor a fixed frontend guarantees retention of earlier predictions:
both expert updates and changed routing can alter them.

## Routing and Stage Decisions

The sample router pools shared complex-feature magnitudes and uses one
`Linear(2c, E)` layer: with `c=16`, its input is 32 dimensional. Per-sample
Top-2 routing evaluates selected independent residual experts and normalizes
their selected weights. No sample is dropped. Expert structure stays fixed;
growth changes expert count and router output width. Parameter memory grows
with expert count, and sparse routing does not guarantee constant wall time.

The stage controller is a transparent validation rule, not a learned policy.
Its novelty statistic compares the current training batch's frozen-frontend
descriptor mean with the fixed stage-one reference, using variance scaling.
The reference does not accumulate later stages. There is no per-expert
autoencoder or validated distribution-shift detector in this demo.

For each later stage:

1. Evaluate the deployed model on current validation rows and, when enabled,
   historical validation rows.
2. Train a separate `adapt_candidate` using current optimizer rows and any
   allowed replay rows. Adapt all existing experts at the old-expert rate and
   the router at its own rate, keeping the frontend fixed.
3. If novelty exceeds its threshold or the adaptation candidate fails the
   acceptance rule, trial one extra expert when the expert budget permits.
   Warmup trains only that expert directly on current training rows at the
   new-expert rate. Then jointly adapt old experts, the new expert and the
   expanded router using their respective parameter groups.
4. Accept a candidate only after its current-validation improvement and
   available historical guard pass. An expansion must also improve enough
   over a qualified adaptation candidate. Otherwise retain the deployed
   model. Actions are recorded as `ADAPT_EXISTING`, `ADD_EXPERT` or `REUSE`.

Trials start from separate copies of the same deployed model. Rejected
candidates must leave the deployed state digest unchanged. Every fit records
its `optimizer_groups`, learning rates and parameter-update statistics so
the configured policy can be checked against executed training. A candidate's
new router row can change Top-2 selection
immediately; expansion is not claimed to be function preserving. Warmup
provides a training path for the new expert without relying on random routing
to select it. Regression loss, balance loss, entropy and expert loads are
reported separately; balanced utilization is not proof of useful specialization.
If a trial's new expert receives zero routes on current or historical validation,
record that diagnostic alongside the rejection reason. Such a rejection does
not establish that an additional expert could not help.

The identical fixed test set is evaluated after each deployed stage, starting
with the before-only baseline. Test metrics and predictions are reporting only;
they never enter controller acceptance, epoch selection or preprocessing
fitting. Historical validation is used for model selection and is therefore
not an untouched test set. Current validation selects each candidate's best
epoch; the historical guard is applied to the resulting candidate. That guard
compares RMSE pooled across all earlier validation rows, so an improvement on
one historical stage can mask regression on another. Stage reports retain
`validation_by_stage` separately for each previously seen stage, including
the original before validation set; these per-stage errors are diagnostics,
not additional acceptance vetoes.

## Replay and Historical Validation

| Configuration | Earlier optimizer rows | Earlier validation rows |
|---|---|---|
| `replay_mode="full"`, `replay_size=0` | All earlier optimizer-training rows enter each candidate adaptation fit | Governed independently by `history_validation` |
| `replay_mode="buffer"`, positive `replay_size` | Reservoir samples may enter candidate optimization | Governed independently by `history_validation` |
| `replay_mode="none"`, `replay_size=0` | No old rows enter candidate optimization | Governed independently by `history_validation` |
| `history_validation=true` | Unchanged replay policy | Used for candidate guards and reports, never optimizer updates |
| `history_validation=false` | Unchanged replay policy | Historical guard is `NOT_RUN`; old-domain retention is unverified |

Full replay and the replay reservoir contain transformed training inputs and
labels, with stage/source identities. Both exclude validation/test rows.
Direct new-expert warmup uses only current-stage training rows.
Reported cumulative training counts count unique rows, not replay presentations.
Disabling replay
while retaining historical validation does not mean that no historical data
are available. The small demo prepares its stages in memory; it does not claim
a streaming memory bound or deletion of previously loaded arrays.

## Artifacts and Scope

| Path relative to `output/data/<run_id>/` | Contents |
|---|---|
| `request.json`, `resolved-config.json`, `manifest.json` | Submitted/resolved settings and frozen source identity |
| `report.json`, `logs/` | Worker phases, status and command logs |
| `reports/demo/demo-report.json` | Stage decisions, metrics, guards, per-fit optimizer groups/update statistics and evidence scope |
| `reports/demo/stages.csv` | Stage-level summary |
| `reports/demo/data-metadata.json` | Real-input physical axes, splits and fitted preprocessing |
| `reports/demo/predictions-stage-001.csv`, etc. | Fixed-test identities, window times, truth, predictions and signed error for each deployment |
| `reports/demo/checkpoint-stage-001.pt`, etc. | Restorable deployment after each stage |
| `reports/demo/checkpoint.pt` | Final deployment artifact |
| `reports/demo/figures/stage-learning-provenance.json` | Figure source hashes, displayed values, transformations and output hashes |
| `artifact-manifest.json` | Exported artifact sizes and SHA256 hashes |

`stage-learning.png` and `stage-learning.pdf` are delivered under
`output/figure/<run_id>/reports/demo/figures/`. They show fixed-test MAE/RMSE
against cumulative after-side training rows, before-validation RMSE on the
same axis, initial/final test predictions versus window time, and the deployed
expert counts/actions. The zero-after-data point is the before-only baseline.
Synthetic or other modes with zero after-side counts use stage index instead.
If timestamps are absent, prediction points use source IDs. Points are never
joined across missing windows; unavailable historical validation remains a gap.
The earlier `before_then_after` distance-group runs also export `distance-learning.png/.pdf` and a
`range-groups.csv` table with per-stage overall and group errors. The table is
under the data report's `figures/` directory; images use the figure delivery root.
Their membership checks, source hashes and output hashes share the stage figure
provenance file.

The renderer
[`plot_stages.py`](../../scripts/MoE_learn/visualization/plot_stages.py)
reads only the saved report and prediction CSVs, checks identical test identities,
truth and times across stages, and recomputes MAE/RMSE to check the recorded
metrics. It converts kilometres to metres for display and applies no smoothing
or interpolation. These single-run plots have no uncertainty intervals.
Rendering runs on haiqin; viewing the retrieved PNG/PDF does not rerun models.

Checkpoint restoration requires this pack, the complete input/normalization
contract, expert IDs, model dimensions and deployment-state identity. Saved
configuration preserves the learning-rate policy. Restoration reproduces
deployment predictions; continuing interrupted training is not implemented.
Dormancy, wakeup, pruning, a learned controller, adaptive frontend growth,
threshold calibration and formal accuracy/cost comparisons are also outside
this first demo.

Synthetic quick PASS covers only the reported CUDA/model/data-fixture
contracts. A short real-data demo or smoke PASS verifies execution and artifacts.
The before-to-after run measures one specified same-record adaptation sequence;
it does not by itself establish a continual-learning advantage, an expert-count
scaling law or unknown-environment robustness. Consult each run's report for
checks actually executed; an earlier run does not validate later source changes.
