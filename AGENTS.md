# AGENTS.md

Workspace instructions for agents working in SwellEx.

## Scope and Execution

Range regression on SwellEx-96 VLA data. Python generates RBD/SCM HDF5 features;
the single active pack `base_pack` trains `complex_CNN`/`ResNet18`/`ResNet50`
regressors and a standalone GRNN per feature. GRNN uses `build`/`predict`, with no
epochs or optimizer. ELM, `real_CNN`, `joint_feature_pack`, `data_scale_pack`,
`MoE_data_scale` and `MoE_learn` are retired under `archive/` (code, docs,
notebooks and results; see `archive/README.md`). They are not importable from
`scripts/`; do not reintroduce them without moving them back deliberately.
The public entry requires `./scripts/run.sh base_pack ...`.
PACK alone starts its configured experiment and follows progress; disconnection does
not stop a launched worker. Select with `--preset`, `--method`, and `--network`;
use `--help` without launching work. Keep shell argument rules in
`scripts/remote_workflow/pack_cli.py`; TOML presets live in
`scripts/remote_workflow/presets/{full,train_before,train_after}.toml`.

Edit code/configuration/docs locally. Run data generation, numerical tests,
training, prediction, and plotting on SSH host `haiqin`. Use the frozen uv
environment; keep `scripts/data_processing/matlab_reference/` static and do
not execute MATLAB. Follow [remote validation](doc/remote_workflow/run-and-validate.md)
for environment setup and command scope. From the local project root:

```bash
./scripts/run.sh base_pack validate --suite smoke
```

The dispatcher uses only the standard library. Snapshot uncommitted/new source
for validation; never overwrite haiqin's existing checkout or original inputs.
Reject silent CPU fallback in the validation workflow. If CUDA fails only
inside a sandbox, recheck outside it before changing drivers/packages.

## Architecture Contracts

- Code roots are `scripts/data_processing/`, `scripts/network_training/`, and
  `scripts/remote_workflow/`; modules sit directly in them (there is no
  per-pack directory layer; `base_pack` is only the pack/run identity). Use
  canonical `scripts.data_processing.*`, `scripts.network_training.*`, and
  `scripts.remote_workflow.*` imports; do not add compatibility aliases for
  removed roots or paths, and preserve strict layer isolation
  (`data_processing` never imports PyTorch/matplotlib/training/workflow;
  `network_training` consumes HDF5 only and never imports `scipy`/`Origindata`/`remote_workflow`).
  `network_training/` holds `neural/` (the single RBD/SCM runtime), `grnn/`,
  `common/` (paths, HDF5 helpers, argument types shared by both) and the
  stdlib-only `catalog.py`. Reuse `common/h5_utils.py` for HDF5 paths and splits;
  feature `METHODS` and axis identities come from `data_processing/input_contract.py`.
- The neural entry is `network_training/neural/main.py {train,predict} --method RBD|SCM`;
  the GRNN entry is `network_training/grnn/main.py {build,predict} --method RBD|SCM`.
  Public shell method names and output directories remain uppercase. Use
  `MethodBindings` (`neural/bindings.py`) for shared train/predict dependencies;
  RBD and SCM differ only by dataset class and model config dataclass.
- RBD `/X` is `[sample,element,frequency,real_imag]`. SCM uses
  `[sample,pair,frequency,real_imag]` with pairs `i<=j`. Pair indices are stored
  1-based in column-major order.
- `neural/model.py:MODEL_REGISTRY` maps registry names (`catalog.neural_model_name`:
  RBD unprefixed, SCM `scm_`) to config and model classes; architectures live in
  `neural/models/`. RBD config uses `input_elements` plus `input_freq_bins`; SCM
  uses `input_pairs` plus `input_freq_bins`. Models return normalized `[batch,1]`.
  Keep GRNN outside trainable model registries.
- Keep pack identity in run/artifact metadata. Never silently reuse a
  checkpoint/reference or fetch a run for another pack. Runs recorded before the
  cleanup (15-model plan with ELM and `real_CNN`) are historical evidence only and
  are rejected by the active plan validation.
- Preserve `DatasetBundle`: `train_dataset`, `val_dataset`, `train_idx`,
  `val_idx`, `train_labels`, `input_shape`, `split_mode`, `source_paths`.
  Its `input_signature` compares frequency, element/depth, and pair axes.
  Current checkpoints/references require this contract; reject missing fields.
- Generation configuration is defined in `generation_config.py`; reject
  unknown/invalid fields. Adaptive frequencies fit valid training windows only.
- Timestamped network exports use `MMDD_HHMMSS`. Resolve latest matching
  artifacts by mtime through `network_training/common/paths.py:latest_time_suffixed_path`.
- Update `pyproject.toml` and `uv.lock` together. Local delivery is
  `output/data/<run_id>/` for logs/tables/reports and `output/figure/<run_id>/`
  for figures. Retired results live in `archive/output/{data,figure}/<category>/`
  (untracked) with an index in `archive/output/README.md`. Remote intermediate
  artifacts use the run's `outputs/`.

## Git and Required Reading

The Git whitelist tracks README/AGENTS, doc, scripts, simulation, notebooks,
`archive/` (code, docs and index files only) and uv project files. Never commit
`Origindata/`, `outputs/`, `output/`, `archive/output/` bulk results, local
environments, macOS junk (`._*`, `.DS_Store`), or stray files. Commit only on user
request; write detailed Chinese commit messages. Push only when requested.

Before changing sensitive areas, read:

- [HDF5 datasets and splits](doc/data_processing/hdf5-datasets-and-splits.md):
  generation, windows, frequency selection, and stored fields.
- [Python extension guide](doc/network_training/extension-guide.md):
  loaders, model registration, and interface constraints.
- [Workflow guides](doc/README.md#remote-workflow): execution and result contracts.
- [Project progress](doc/project-progress.md): verified scope and gaps.
- [Experiment pack](doc/README.md#experiment-pack): pack ownership and commands.
- [Archive index](archive/README.md): what was retired and how to restore it.

Keep the language of the document being edited. Check current source and
reports before stating behavior or scientific validation claims.
