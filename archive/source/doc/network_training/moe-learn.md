# Continual MoE Staged Learning and Candidate Growth

`scripts/network_training/MoE_learn/` (`demo.py`, `common/controller.py`, `visualization/plot_stages.py`) evaluates staged `RBD_SCM` range regression (`complex_MoE_learn`, fixed `block_rms`), adapting existing experts at a reduced learning rate while validating candidate expert expansion. Staged/holdout dataset splits (`distance_holdout`, `before_then_after`, `interleaved`, `chronological`) and per-window mean centering live in [HDF5 datasets and splits](../data_processing/hdf5-datasets-and-splits.md#continual-stage-and-distance-holdout-splits); shell commands and TOML presets live in [remote validation](../remote_workflow/remote-validation.md) and [configuration defaults](../remote_workflow/command-defaults.md#continual-moe-staged-defaults).

## Stage-1 Freezing and Grouped Optimization

Only **stage 1 optimizer-training rows** (fitted strictly after any `distance_holdout` exclusions) determine the two `block_rms` scales, target mean/SD, and frontend descriptor reference; the `c, 2c` complex frontend is frozen in eval mode after stage 1.

Per-sample Top-2 routing (`Linear(2c, E)` on pooled shared magnitudes) dispatches to independent `2c -> 4c` GroupNorm residual experts without dropping samples. Deployment checkpoints freeze expert gradients for inference; candidate trials clone independent copies of the deployed model and re-enable expert/router gradients. Positive finite learning rates require `old_expert_learning_rate < new_expert_learning_rate`, with an independent `router_learning_rate`; optimizer groups, rates, and parameter-update norms are persisted at every stage.

## Candidate Decisions, Replay, and Historical Guards

At each stage $t \ge 2$:

1. Evaluate current deployment on stage-$t$ validation and (when `history_validation=true`) pooled earlier validation rows.
2. Train `adapt_candidate` on stage-$t$ training + allowed replay using `old_expert` and `router` LR groups.
3. If novelty exceeds `novelty_threshold` or adaptation fails acceptance (and `E < max_experts`), warm up 1 new expert directly on stage-$t$ training rows only (`new_expert` LR), then jointly adapt old experts, new expert, and expanded router (`old_expert`, `new_expert`, and `router` groups). Accepted new experts become `old_expert` at stage $t+1$.
4. Accept `ADAPT_EXISTING` or `ADD_EXPERT` only if current validation improves by at least `min_relative_gain` and pooled historical validation does not regress beyond `max_old_relative_regression` (otherwise keep deployment via `REUSE`).

| Setting | Scope |
|---|---|
| `replay_mode = "full" \| "buffer" \| "none"` | Controls whether all, reservoir-sampled, or zero prior **training** rows enter adaptation fits (never warmup) |
| `history_validation = true \| false` | Independently enables pooled historical **validation** guards; per-stage errors are always logged in `validation_by_stage` because pooled RMSE can mask single-stage regression |

Test rows are evaluated after every stage for reporting only and never select actions or epochs.

## Checkpoints, Plots, and Scope

Stage exports write `demo-report.json`, `stages.csv`, `data-metadata.json` (including persisted excluded-before identities for `distance_holdout`), `predictions-stage-*.csv`, `checkpoint-stage-*.pt`, and final `checkpoint.pt`; `plot_stages.py` renders `stage-learning.{pdf,png}` (plus `distance-learning.{pdf,png}` and `range-groups.csv` for `range_bins`). Checkpoint restoration reproduces deployment inference after verifying pack, architecture, preprocessing, stage membership, and holdout identities; mid-stage training resume, automatic expert pruning, and retention guarantees are out of scope. Recorded runs are summarized in [project progress](../project-progress.md).
