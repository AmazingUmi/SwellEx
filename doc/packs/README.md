# Experiment Packs

A pack owns an experiment's generation, input representation, models, remote
workflow, and visualization code. Select it explicitly before the command:

```bash
./scripts/run.sh PACK [options]
./scripts/run.sh PACK METHOD ARCH [options]
./scripts/run.sh PACK validate --suite smoke [options]
./scripts/run.sh PACK status --run RUN_ID [options]
./scripts/run.sh PACK fetch --run RUN_ID [options]
```

`PACK` alone runs the full experiment, equivalent to
`PACK validate --suite experiment`. The base plan contains 15 models; the
joint plan contains 35 for one fusion mode. Bundled settings use the full S5
recording and 100 epochs per neural model. GRNN uses build/predict. Explicit
`validate` retains its `smoke` default; `PACK --suite quick|smoke|experiment`
can also select a suite.

```bash
./scripts/run.sh joint_feature_pack --dry-run
./scripts/run.sh joint_feature_pack --fusion raw_concat --detach
./scripts/run.sh base_pack --config /tmp/experiment.toml
```

Pack-only options are the validation options: `--config`, `--suite`,
`--dry-run`, `--detach`, `--no-cache`, and joint `--fusion`. Set experiment
epochs/seed in the TOML; the `--epochs`/`--seed` flags apply to `METHOD ARCH`.
`PACK --help` displays the pack's commands and model choices without loading
configuration or launching work. Unknown, incomplete, and repeated options
are rejected. Paths with spaces are passed unchanged when quoted.

| Pack | Purpose | Feature selections |
|---|---|---|
| [base_pack](base-pack.md) | Preserve the original independent-feature experiment | `RBD`, `ELM`, `SCM` |
| [joint_feature_pack](joint-feature-pack.md) | Compare combinations of existing features with existing model architectures | Three singles, three pairs, and the three-way combination |

Both packs support `complex_CNN`, `real_CNN`, `ResNet18`, `ResNet50`, and
standalone `GRNN`. Pack and method names are case-sensitive. An omitted or
unknown pack is rejected; the old `./scripts/run.sh RBD complex_CNN` form is
not an alias. `--config` and all other options follow the pack argument.

Each pack has its own `data_processing/`, `localization_networks/`, and
`visualization/` under `scripts/<pack>/`. Imports use the same pack namespace.
The joint pack starts from the original Python implementation; experimental
changes belong there. `base_pack` retains the original feature definitions,
split semantics, and 15-model plan. The static MATLAB reference is preserved
only in `scripts/base_pack/data_processing/matlab_reference/` and is not run.

## Configuration and Results

Each pack's complete TOML presets live under
`scripts/<pack>/localization_networks/workflow/`:

- `haiqin_full_interleaved.toml`: the default full-record interleaved split.
- `haiqin_range_overlap_train_before.toml`: before-to-after shared-range task.
- `haiqin_range_overlap_train_after.toml`: after-to-before shared-range task.

Use a preset belonging to the selected pack. The joint pack adds its fusion
setting; the base pack rejects that extra setting. A dry run prints the frozen
request without starting computation or SSH:

```bash
./scripts/run.sh base_pack SCM ResNet18 --dry-run
./scripts/run.sh joint_feature_pack RBD_ELM_SCM ResNet18 --fusion block_rms --dry-run
```

Generation, numerical tests, training, prediction, and plotting run on haiqin
using the frozen uv environment. Source snapshots include uncommitted/new
files. Follow the [remote workflow](../workflows/remote-validation.md) for
execution and recovery.

Delivery remains `output/data/<run_id>/` and `output/figure/<run_id>/`; each
run records its pack in the request, report, and source manifest. Use the same
pack for status and fetch; a mismatched pack is rejected. Keep pack, split,
feature combination, fusion, and seed explicit when comparing results.

## Evidence Boundaries

Runs recorded before the migration in [project progress](../overview/project-progress.md)
establish the original scheme's recorded
results, not validation of the moved code or a joint-feature benefit. New
checks must identify their current source snapshot and pack. Pipeline PASS
establishes execution and contract checks; accuracy claims require comparable
test identities, repeated seeds, and a defined comparison protocol.
