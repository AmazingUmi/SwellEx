# SCM Data-size and Network-capacity Study

Historical protocol: v1 training artifacts and figures were removed during the
2026-10-16 retention cleanup. The current complete record is
[the controlled-budget study](data-scale-budget-study.md#completed-results-2026-10-16);
its frozen calibration retains the development validation projection.

## Material Passport

- Origin: experiment-agent run workflow; user-authorized experimental design.
- Date: 2026-10-15.
- Protocol: `scm-capacity-study-v1`; frozen before matrix execution.
- Verification status: protocol only; execution status belongs to `study.json`.

## Question and Controls

Does the preferred complex CNN depth depend on the number of optimizer-training
windows, and how does the frozen training-count rule compare with fixed depths?
SCM is the primary input so feature representation stays constant. This study
does not establish the same relationship for RBD/ELM or new environments.

| Factor | Frozen setting |
|---|---|
| Recording / features | S5, 0–4499 s, 1 s disjoint windows, deep+shallow frequencies, SCM |
| Outer split | `full_interleaved` 4:1; 900 fixed test windows |
| Validation | 25% of 3,600 outer training rows; 900 rows per seed |
| Retained fractions | 0.125 / 0.25 / 0.5 / 1.0 |
| Expected actual training counts | 337 / 675 / 1,350 / 2,700; checked from physical identities |
| Seeds | 2026 / 2027 / 2028; initialization, shuffle, subset and validation split vary together |
| Networks | auto; fixed small / medium / large / xlarge; original baseline |
| Optimization | 100 epochs, batch 64, AdamW lr 0.0003, weight decay 0.0001, cosine to zero |
| Model / loss | base channels 16, dropout 0.15, SmoothL1 in km with beta 0.5 |
| Automatic rule | reference_samples=600, exponent=0.5; unchanged, uncalibrated |
| Matrix | 12 child runs, 72 trained models; pilot results excluded |

The full-data xlarge pilot used 10 then 100 epochs in separate runs. Validation
error continued improving after 10 epochs. The 100-epoch pilot's final part was
much flatter, so 100 epochs is the common comparison budget, not a claim that
every model/data-size pair has converged. Best-checkpoint epoch and full curves
are retained to expose training-budget sensitivity.

Training subsets are nested within each seed; all profiles at a given seed and
fraction use identical physical training/validation members. Every run uses
the same test members. No held-out labels determine training subsets. All input
and target normalization retain the pack's existing contracts. `adapt` is
rejected; inherited whole-record mean subtraction remains an explicit boundary.

## Analysis and Figures

The primary development metric is validation RMSE at the saved best checkpoint.
Report test RMSE and MAE for that checkpoint, along with actual parameter count,
block count, epoch time, training-process wall time and optimizer steps. Keep
the original baseline distinct from the scalable four-block medium network.

Show every seed and the mean with sample SD (`ddof=1`). The three seeds describe
combined initialization/split/subset variability on one recording; they are not
independent environmental replicates and SD is not a confidence interval.
Choose a descriptive fixed-profile reference by mean validation RMSE, with
ties resolved toward fewer parameters; test scores never choose this reference
or adjust the automatic rule. This development comparison is not an independently
validated capacity predictor. A fixed small model winning throughout is valid.

Required outputs: validation-error heatmap over N and capacity; test learning
curves over N; error versus parameter count; online-training and validation
epoch trajectories; training time versus error; and actual selected capacity
versus N. CSV/JSON tables preserve per-seed observations, selection rules,
missing runs and provenance. PDF and PNG figures use shared complete legends.

All models see the same number of epochs, not the same number of optimizer
updates. More data increases both information and work per epoch; plot actual
steps/time and interpret the result as performance under a fixed-pass budget.
Online training metrics include dropout and changing weights and are not a
strict post-training generalization-gap estimate. No inferential significance
test or optimal scaling-law claim is planned from three seeds.

## Execution and Evidence

Use the [matrix command](data-scale-pack.md#data-size-matrix). Numerical tests,
training, prediction, aggregation and plotting execute on haiqin with frozen uv
and isolated source snapshots. Keep the original data and checkout unchanged.
Do not mix interrupted/interfered performance measurements with clean runs.
Source/config hashes, complete sample membership, histories and delivery hashes
must verify before combining results. Failed/missing cells stay explicit.

The numerical-run lock serializes supported workloads. Observed times belong
to this GPU/runtime and are not dedicated-hardware benchmark guarantees.
`full_interleaved` is within-record interpolation among correlated windows;
cross-segment and independent-environment studies remain separate work.

Scientific critical-thinking and visualization procedures were assisted by
Scientific Agent Skills: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner,
A. M. (2026), [Scientific Agent Skills: A Library of Procedural Knowledge for
Research Agents](https://doi.org/10.48550/arXiv.2609.00065). This credits workflow
assistance, not evidence for the network-capacity hypothesis.
