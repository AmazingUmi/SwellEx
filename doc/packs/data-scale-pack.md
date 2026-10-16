# Data Scale Pack

`scripts/data_scale_pack/` tests whether training sample count can choose a
complex CNN's depth before optimization. RBD, ELM and SCM remain separate
inputs. This pack has no feature fusion or GRNN. The rule is **uncalibrated**;
engineering PASS does not establish optimal capacity or generalization.

## Commands

```bash
# All three inputs and all six network choices: 18 models, 100 epochs each.
./scripts/run.sh data_scale_pack --preset train_before

# One training-count-selected network.
./scripts/run.sh data_scale_pack --preset train_before --method RBD --network complex_CNN

# Inspect the interface without launching computation.
./scripts/run.sh data_scale_pack --help

# Query or retrieve a child run, or the study ID printed after Run:.
./scripts/run.sh data_scale_pack status --run 'RUN_ID'
./scripts/run.sh data_scale_pack fetch --run 'RUN_ID'
```

The shared [remote workflow](../workflows/remote-validation.md) applies, including
frozen source, CUDA, status/fetch and delivery directories. Use the launching
pack in status/fetch; descriptive child IDs contain `scale`, and study launches
print their registered study ID. Pack alone selects the
full experiment with the `full` preset. `validate --suite smoke` selects auto
and baseline for each input (6 models); `quick` tests all 18 combinations and
contracts. Smoke uses the configured interval and epochs; it does not shorten
them automatically.

## Network Family

| `--network` | Stage repetitions | Selection |
|---|---|---|
| `complex_CNN` | Chosen before model initialization | Training-count rule |
| `complex_CNN_small` | `[1,1,1]` | Fixed, 3 blocks |
| `complex_CNN_medium` | `[1,2,1]` | Fixed, 4 blocks |
| `complex_CNN_large` | `[2,2,2]` | Fixed, 6 blocks |
| `complex_CNN_xlarge` | `[2,3,3]` | Fixed, 8 blocks |
| `complex_CNN_baseline` | Original four blocks | Preserved architecture control |

Each scalable stage has `c`, `2c`, or `4c` complex channels (`c=16` by default).
Stage-entry strides are `(1,2)`, `(2,2)`, `(2,1)` along feature rows/frequency;
additional blocks use `(1,1)`. Width, downsampling, normalization and regression
head are fixed across the four scalable profiles. There are no residual links.
The primitive is two-kernel complex convolution, separate real/imaginary batch
normalization, and modReLU. Final magnitude and adaptive average pooling feed
the same two-layer real regression head.

The baseline retains the original widths `c,2c,4c,4c` and strides
`(1,2),(2,2),(2,2),(1,2)`. It therefore differs from the scalable medium profile
despite both having four blocks. Outputs remain normalized `[batch,1]`.

## Training Count and Capacity Rule

`N` counts valid unique S5 windows actually used for optimizer updates, after
validation splitting and optional training subsampling. Validation/test rows,
feature dimensions, epochs and repeated batches do not increase `N`. Missing
source IDs/times or repeated S5 source IDs reject; no invented row IDs are used.

For actual trainable parameter counts `P_small`, `P_medium`, and `P_xlarge`:

```text
budget = clip(P_medium * (N / reference_samples) ** exponent,
              P_small, P_xlarge)
profile = largest scalable profile whose parameter count <= budget
```

Counts include both real kernels, normalization/activation parameters and the
regression head. Fixed controls ignore the budget when choosing their profile,
but record the same budget for comparison. The defaults `reference_samples=600`
and `exponent=0.5` are starting hypotheses, not fitted scaling-law coefficients.
The policy version is `train-count-v1`.

Copy a bundled complete TOML to change data amount or policy:

```toml
# Within the existing [training] table:
train_fraction = 0.5

[scaling]
reference_samples = 600
exponent = 0.5
```

`train_fraction` must be in `(0,1]` and retain at least two training rows;
`reference_samples` must be a positive integer and `exponent` finite and positive.
All tables/fields are strict; no GRNN/fusion settings are accepted.
Frequency selection supports fixed `deep`, `shallow`, `mel`, and `full` modes.
`adapt` is rejected because the current generator would fit frequencies before
validation splitting and training subsampling, exposing excluded observations.

The validation split is established first and stays fixed when changing only
`train_fraction`. Training candidates are time-sorted into blocks of 32 rows;
a seeded block shuffle defines a nested prefix, with only the final block
possibly truncated. The retained count is `floor(candidate_count*fraction)`.
This does **not** establish temporal independence, guarantee equal distance
coverage, or replace the existing random training/validation split. Inspect
distance coverage during calibration and keep window geometry fixed. Increasing
overlap or duplicating observations must not be interpreted as new information.
The inherited generator still subtracts the full recording mean, including
held-out times. The retained count measures optimizer-training rows, not every
raw observation accessed during preprocessing.

## Artifacts and Calibration

Before the first optimizer step, `*_architecture_MMDD_HHMMSS.json` records the
pack/method/model, complete resolved configuration, selected profile, stage
layout, parameter count/budget and training subset metadata. Checkpoints also
retain full training/validation source SHA256 identities and exact indices.
Resume rejects changed data, selection, width or capacity policy. Prediction
rebuilds the saved architecture; test sample count never selects a new model.
Reports and visualization CSVs retain the realized profile/count/parameters.

For calibration, compare all fixed scalable profiles and auto at several
training fractions and seeds. Within each seed, keep validation/test identities
identical across fractions and profiles. Changing the seed also changes the
validation split; the generated test identities remain fixed. Choose rule
coefficients using development validation only, freeze them, then evaluate
on held-out conditions. Track error, bias, parameter count, training cost and
convergence. A fixed profile winning at every data size is a valid negative
result. Cross-segment S5 evidence is not independent-environment generalization.

Recorded checks and unfinished scientific work belong in
[project progress](../overview/project-progress.md).

## Data-size Matrix

The next [calibration and controlled-budget study](data-scale-budget-study.md)
uses `budget-study --plan ...`, fresh seeds and equal-update controls. Its
frozen calibration is an experiment artifact; it does not replace the default
capacity policy. Controlled training rejects resume and records full batches,
optimizer updates and validation checkpoints explicitly.

The [SCM study protocol](data-scale-study.md) freezes four fractions, three seeds
and all six network choices, with 100 epochs per model:

```bash
./scripts/run.sh data_scale_pack study \
  --plan scripts/data_scale_pack/localization_networks/workflow/scm_capacity_study_v1.json \
  --study-dir output/data/scm-capacity-study-v1
./scripts/run.sh data_scale_pack status --run 'scm-capacity-study-v1'
./scripts/run.sh data_scale_pack fetch --run 'scm-capacity-study-v1'
```

Both matrix commands start the coordinator on haiqin, print `Run: STUDY_ID`, and
return. It runs child jobs serially and saves `study.json` with the complete plan,
source inventory and run IDs. The matrix continues after the local terminal
disconnects. `status` checks the remote coordinator and children; `fetch` retrieves
available evidence and child deliveries through the same public entry.

Repeating the launch command recovers that study. Completed children keep their
original results, an active child is followed, and only untouched children start.
Failed or ambiguous attempted children are never automatically retrained. A
previous local ledger can be imported with its original frozen training source;
coordinator migration evidence is separate. Plan or numerical-source changes
reject continuation. Cross-run analysis verifies exact physical memberships and
uses actual training counts rather than nominal fractions.

On haiqin, render the completed ledger with the frozen interpreter and source:

```bash
python -B -m scripts.data_scale_pack.visualization.plot_study \
  --study /path/to/study.json --runs-root /path/to/validation/runs \
  --output-dir /path/to/new-analysis
```

The renderer verifies raw metrics and membership before producing CSV/JSON and
six PDF/PNG figure sets. A later layout revision requires explicit
`--allow-renderer-update` and a new output directory; its manifest records the
actual renderer hashes, differences from the training snapshot, and that no
training was repeated. Local delivery separates tables/reports under
`output/data/<study_id>/` and figures under `output/figure/<study_id>/`.
Child deliveries are nested in each study directory's `children/<run_id>/`;
the same layout applies to direct child fetch when its study ledger is local.
Study readers default to the adjacent `children/` directory. Pass `--runs-root`
explicitly when reading haiqin's separate execution directories.
