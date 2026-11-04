# Controlled Matrix Studies

`scripts/remote_workflow/data_scale_pack/` (`study`, `budget-study`) and `scripts/remote_workflow/MoE_data_scale/` (`study`) run detached multi-seed matrix coordinators on `haiqin`. Re-running a launch command imports existing ledgers, preserves completed children, follows active ones, and never silently retrains failed cells. Study tables/reports land in `output/data/<study_id>/` and figures in `output/figure/<study_id>/`, with child runs nested under `output/{data,figure}/<study_id>/children/<run_id>/`.

## SCM Capacity Calibration and Budget Study

Model profiles and the `train-count-v1` capacity rule live in [data scale pack](../network_training/data-scale-pack.md).

```bash
./scripts/run.sh data_scale_pack budget-study \
  --plan scripts/remote_workflow/data_scale_pack/scm_capacity_budget_study_v2.json --dry-run
./scripts/run.sh data_scale_pack budget-study \
  --plan scripts/remote_workflow/data_scale_pack/scm_capacity_budget_study_v2.json \
  --study-dir output/data/scm-capacity-budget-study-v2
./scripts/run.sh data_scale_pack status --run 'scm-capacity-budget-study-v2'
./scripts/run.sh data_scale_pack fetch --run 'scm-capacity-budget-study-v2'
```

- **Development calibration**: Fitted on `scm_capacity_study_v1` (S5 0–4499 s, `full_interleaved` 4:1, 900 fixed test + 900 validation windows, nested fractions `0.125/0.25/0.5/1.0` giving `N = 337/675/1350/2700`, seeds `2026–2028`, 100 epochs). Using **validation** RMSE only across the four scalable profiles (excluding test metrics and baseline), a 338-point grid (`reference_samples=600..1800`, `exponent=0.25..1.50`) minimized equally weighted relative validation regret across `N`, breaking ties toward lower mean parameter count and larger log-margin to switching thresholds. Frozen `scm_capacity_calibration_v2.json` (SHA256-verified) selected `reference_samples=900, exponent=0.9`, mapping `337/675/1350/2700` to `small/small/medium/large` (thresholds `900`, `~2083.8`, `~3571.3` rows); default `600 / 0.5` stays unchanged in code.
- **Fresh-seed controlled comparison**: Disjoint seeds **2029/2030/2031** train the four scalable profiles plus `complex_CNN_baseline` under batch 64 (drop tail, continuous seeded stream, `--resume` rejected, per-update cosine LR to zero, 100 validation checks): **Epoch arm** runs 100 passes (`500 / 1000 / 2100 / 4200` updates); **Update arm** runs fixed `4200` updates (`268,800` presentations). Both arms coincide at `N=2700`, sharing execution (**21 child runs / 105 trainings** for **24 cells / 120 observations**).
- **Completed results (2026-10-16)**: All 21 children passed (`output/data/scm-capacity-budget-study-v2/analysis-v2/` and `output/figure/scm-capacity-budget-study-v2/v2/`). Mean ± sample SD (`ddof=1`, 3 seeds) test RMSE (m) for the validation-selected scalable profile:

| Training `N` | Validation-selected profile | 100 passes (m) | 4200 updates (m) |
|---:|---|---:|---:|
| 337 | `small` | 1090.66 ± 133.32 | 1056.14 ± 74.94 |
| 675 | `small` | 730.93 ± 42.88 | 688.28 ± 23.01 |
| 1350 | `small` | 575.04 ± 35.66 | 536.57 ± 19.72 |
| 2700 | `large` | 358.46 ± 5.90 | Shared execution |

Re-render a completed data-scale ledger on `haiqin` via `python -B -m scripts.remote_workflow.data_scale_pack.visualization.plot_study --study /path/to/study.json --runs-root /path/to/validation/runs --output-dir /path/to/new-analysis`.

## RBD-SCM MoE Expert Study

Residual MoE architecture and routing contracts live in [MoE data scale](../network_training/moe-data-scale.md).

```bash
./scripts/run.sh MoE_data_scale study \
  --plan scripts/remote_workflow/MoE_data_scale/rbd_scm_expert_study_v1.json --dry-run
./scripts/run.sh MoE_data_scale study \
  --plan scripts/remote_workflow/MoE_data_scale/rbd_scm_expert_study_v1.json \
  --study-dir output/data/rbd-scm-moe-expert-study-v1
./scripts/run.sh MoE_data_scale status --run 'rbd-scm-moe-expert-study-v1'
./scripts/run.sh MoE_data_scale fetch --run 'rbd-scm-moe-expert-study-v1'
```

`rbd_scm_expert_study_v1.json` runs 16 conditions / **48 trainings**: S5 `train_before` (700 shared test windows, 200 validation windows fixed per seed), nested fractions `0.125 / 0.25 / 0.5 / 1.0` (`75 / 150 / 300 / 600` training windows) × fixed `E1 / E2 / E4 / E8` × seeds `2026 / 2027 / 2028`, 100 epochs (`200 / 300 / 500 / 1000` updates), batch 64, `c=16`, and `balance_loss_weight=0.01`, writing `study-cells.csv`, `study-summary.csv`, `validation-selection.csv`, and `study-report.json` under `output/data/<study_id>/`. Interpretation boundaries for both matrices live in [project progress](../project-progress.md).
