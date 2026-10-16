# SCM Capacity Calibration and Training-budget Study

## Run and Recover

```bash
./scripts/run.sh data_scale_pack budget-study \
  --plan scripts/data_scale_pack/localization_networks/workflow/scm_capacity_budget_study_v2.json \
  --dry-run
./scripts/run.sh data_scale_pack budget-study \
  --plan scripts/data_scale_pack/localization_networks/workflow/scm_capacity_budget_study_v2.json \
  --study-dir output/data/scm-capacity-budget-study-v2

./scripts/run.sh data_scale_pack status --run 'scm-capacity-budget-study-v2'
./scripts/run.sh data_scale_pack fetch --run 'scm-capacity-budget-study-v2'
```

The launch command prints `Run: scm-capacity-budget-study-v2` and returns after
starting the coordinator on haiqin. The entire matrix, including subsequent
children, continues without the local terminal or computer. `status` queries the
remote coordinator and child reports; a stopped coordinator is not reported as
healthy `RUNNING`. `fetch` retrieves the ledger, logs and available child results
once; it does not wait for the remaining children.

Repeat the launch command to recover the same remote study. A local frozen ledger
from the earlier coordinator can be imported: completed children retain their
original run IDs and results, active children are followed, and only untouched
children are launched. An attempted failed or ambiguous child is never silently
retrained. The frozen plan, numerical source and existing results must verify
before continuation. Coordinator migrations record their separate source
identity without replacing the training snapshot.

Individual children still use `./scripts/run.sh data_scale_pack status --run 'RUN_ID'`
and `./scripts/run.sh data_scale_pack fetch --run 'RUN_ID'`. Use the launching pack;
`joint_feature_pack` cannot manage this study. If the plan changes the remote
host/workspace, pass a matching `--config PATH` to status/fetch. The former local
`status.py` and coordinator wrapper are obsolete; use only `run.sh` for management.

Data, ledgers and reports go to `output/data/scm-capacity-budget-study-v2/`;
figures go to `output/figure/scm-capacity-budget-study-v2/`. Raw HDF5 and weights
remain on haiqin. The short smoke matrix has a separate identity and is excluded.

Each study has one top-level directory in `output/data/` and `output/figure/`.
Its child reports and figures are nested under `children/<run_id>/` inside those
directories. Whole-study fetch and direct child fetch both use that location
when the local study ledger is present. The main `report.md` remains the reading
entry point; child `report.json` files retain the execution details. Remote
execution paths remain `<workspace>/runs/<run_id>/`.

## Development Calibration

Protocol date: 2026-10-15. This extends the completed
[v1 development experiment](data-scale-study.md). Its test results have already
been inspected; the present calibration is explicitly development work, not
blind validation. The fit receives a whitelist projection of **validation**
RMSE only: four fixed profiles × four training counts × seeds 2026–2028.
Neither test metrics, the original baseline, nor duplicate auto runs enter it.

Keep the two-parameter training-count rule and the existing four depths. Search
`reference_samples=600,700,...,1800` and `exponent=0.25,0.30,...,1.50` (338
candidates). For each N, average validation RMSE over development seeds; divide
the selected profile's mean by the lowest four-profile mean and subtract one.
Minimize the equally weighted mean of those four relative regrets. Equal N
weights prevent the larger errors at small N from setting the whole objective.

Scores within 1e-12 of the minimum tie. Prefer lower mean parameter count. Within
each remaining identical N-to-profile mapping, prefer the largest minimum
absolute log-distance from observed N to the three switching thresholds. Then
prefer lower exponent, followed by lower reference count. These choices are
frozen before executing the grid. Report all candidates and three descriptive
leave-one-development-seed-out checks; no confidence interval or significance
test is inferred from them.

The calibration artifact records the validation projection, source/plan/input
hashes, objective, grid, tie rule, policy and thresholds. Freeze its **raw byte
SHA256** in the new plan and every child request. The default `train-count-v1`
policy stays unchanged. The calibrated candidate is specific to this SCM
development protocol; new budgets test its transfer. Profile clipping outside
the observed N range does not establish valid extrapolation.

The executed calibration selected `reference_samples=900, exponent=0.9`, mapping
337/675/1350/2700 to small/small/medium/large. Continuous switching thresholds
are 900, approximately 2083.83 and 3571.27 rows; the final threshold exceeds the
observed range. Multiple coefficients yield the same four assignments, so these
are development-selected thresholds rather than uniquely identified optima.
One leave-one-seed-out fit changes to 1000/1.5, exposing parameter sensitivity.
The exact artifact and its raw byte hash are in the executable plan. It does
not establish superiority under the new controlled protocol.

## Fresh-seed Controlled Comparison

Use the same S5 SCM features, physical splits, 0–4499 s recording, nested fractions
0.125/0.25/0.5/1 and expected training counts 337/675/1350/2700. Seeds are
**2029/2030/2031**, disjoint from calibration. Train the four fixed profiles and
original CNN baseline. The old and calibrated policies select among these exact
fixed-model observations; selecting the same profile references the same result
and creates neither a new training run nor a new replicate.

| Control | Epoch arm | Update arm |
|---|---|---|
| Full batch size | 64, drop incomplete tail | 64, drop incomplete tail |
| Optimization budget | 100 complete data passes | 4200 optimizer updates |
| Updates at the four N | 500 / 1000 / 2100 / 4200 | 4200 at every N |
| Validation | End of each pass, 100 checks | Every 42 updates, 100 checks |
| Learning rate | Per-update cosine over total budget | Per-update cosine over total budget |
| Sample presentations | Updates × 64 | 268800 at every N |

AdamW lr 0.0003, weight decay 0.0001, channels 16, dropout 0.15, and SmoothL1 in
km with beta 0.5 are unchanged. Independent seeded generators isolate training
shuffle and validation loading from model initialization. A continuous shuffled
batch stream crosses validation boundaries without restarting its data pass.
Record initial weights, batch-source sequence, learning-rate sequence, actual
updates, validation locations and consumed source IDs. Every retained member
must actually receive gradient updates before a run passes. Controlled resume
is rejected because exact sampler/RNG restoration is not implemented.

At N=2700 the two schedules coincide: 42 batches/pass × 100 passes. Share the
same trained result across arms only after verifying actual N and budget. Thus
the study has **21 unique children / 105 real trainings**, representing **24
analysis cells / 120 fixed-model observations**. Shared rows and policy rows
must retain source IDs and must not inflate sample size. Numerical tests compare
both budget modes' batches, learning rates, validation and final weights.

The controlled epoch arm differs from v1 in full batches, shuffle isolation and
per-update scheduling. Do not treat old v1 runs as matched causal controls.
Equal updates controls optimization work and sample presentations; unique N,
coverage and repeated use still differ. An epoch-versus-update comparison also
changes the cosine schedule length and validation locations at smaller N, so it
compares complete budget protocols rather than information alone.

## Analysis and Evidence Limits

Plot N, parameters, validation/test RMSE, training time and optimizer updates;
show both arms, every seed, mean and sample SD. Plot trajectories against actual
updates. Select a descriptive fixed-profile reference using fresh-seed mean
validation RMSE within each N and arm, with smaller-model ties. Validation
regret averaged across seeds is nonnegative; individual seed differences can
be negative because the reference minimizes the cross-seed mean. The test
difference against that validation-selected reference is signed and must not
be called test-oracle regret. Never clip negative paired differences.

Compare old and calibrated policies as paired seed/N/arm differences; identical
source results have exactly zero difference. Average across N within each seed
before any cross-N seed summary. Three seeds describe combined split, subset
and initialization variation. Fresh seeds reuse the same physical recording and
test set; they establish seed replication and protocol transfer, not independent
environmental validation. Whole-record mean subtraction and same-record
`full_interleaved` interpolation retain the v1 evidence boundaries.

All numerical work and plotting run on haiqin with frozen uv and isolated source
snapshots. Every child verifies the immutable remote training snapshot; controller
updates have separate provenance. Verify complete member
identities, histories, source/config/calibration and delivery hashes before
aggregation. Missing or failed cells remain explicit. Scientific workflow
assistance is credited in the [v1 protocol](data-scale-study.md).

## Completed Results (2026-10-16)

`scm-capacity-budget-study-v2` completed at 17:44 Singapore time: **21 children /
105 real trainings PASS**, with no missing cells. The remote coordinator continued
the remaining 45 trainings under their original IDs, preserving the 60 completed
results and original numerical source. Aggregation verifies 120 logical model
observations, 48 virtual policy views and 10,500 recorded validation checkpoints;
these aliases do not increase the number of independent trainings.

Among the four scalable profiles, both budgets select small/small/small/large by
mean fresh-validation RMSE. Test values below are mean ± sample SD over three
seeds, in metres; they are not confidence intervals or test-selected minima.

| Training N | Validation-selected profile | 100 data passes | 4200 updates |
|---:|---|---:|---:|
| 337 | small | 1090.66 ± 133.32 | 1056.14 ± 74.94 |
| 675 | small | 730.93 ± 42.88 | 688.28 ± 23.01 |
| 1350 | small | 575.04 ± 35.66 | 536.57 ± 19.72 |
| 2700 | large | 358.46 ± 5.90 | Same shared execution |

The frozen calibrated rule still chooses small/small/medium/large. Against the
original rule, its N=675 switch to small reduces mean test RMSE by 53.70 m in the
epoch arm and 24.58 m in the update arm. Its N=2700 switch to large reduces it by
35.53 m. The other two N choices are identical. Equal-N averaging within each
seed gives improvements of 22.31 ± 14.94 m and 15.03 ± 7.98 m respectively.
These are descriptive paired differences, not significance claims.

Small-data preference for a smaller network persists with equal optimizer updates;
it is not explained solely by the epoch arm's fewer updates. More updates do not
universally improve error: at N=337, large, xlarge and baseline mean test errors
increase. At N=1350, the calibrated medium choice is still above the small
profile's validation mean. The experiment does not identify a universal or exact
switching threshold, and the default policy remains uncalibrated.

At full N, large uses 148,481 parameters and has test RMSE 358.46 ± 5.90 m;
the original baseline uses 125,201 parameters and has 360.93 ± 9.29 m. Their
means are close, and baseline has slightly lower validation RMSE. This does not
establish a clear improvement over the original architecture.

Final tables and provenance are in `output/data/scm-capacity-budget-study-v2/analysis-v2/`,
descriptive diagnostics in `diagnostics/`, and the completion report in `report.md`.
Seven final PDF/PNG figure pairs are in `output/figure/scm-capacity-budget-study-v2/v2/`.
The v2 renderer only lays out heatmap annotations differently: all 11 CSV tables
and `study-results.json` are byte-identical to the first export. Child delivery
verified 273 manifest files; each aggregate export verified 27 files.

Retention cleanup on 2026-10-16 keeps this study and its 21 necessary children,
final `analysis-v2/` tables, diagnostics, v2 figures, frozen sources and referenced
feature cache. Earlier data-scale matrices, smoke runs, pilots, coordinator
helpers and duplicate exports were removed locally and on haiqin. The frozen
calibration artifact retains its 48-row development validation projection and
source hashes; earlier full training artifacts are no longer retained. Final
verification records are consolidated in the study’s `completion.json`.
