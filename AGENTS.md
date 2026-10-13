# AGENTS.md

Workspace instructions for agents working in SwellEx.

## Scope and Execution

Range regression on SwellEx-96 VLA data. Python generates RBD/ELM/SCM HDF5
features; independent PyTorch packages train CNN/ResNet regressors. Standalone
RBD/ELM/SCM GRNN uses `build`/`predict`, with no epochs or optimizer. The original
scheme lives in `base_pack`; joint-feature experiments live in
`joint_feature_pack`. The public entry requires `./scripts/run.sh PACK ...`.
PACK alone starts its full experiment and follows progress; disconnection does
not stop a launched worker. Select with `--preset`, `--method`, and `--network`
(joint also accepts `--fusion`); use `--help` without launching work.
Keep shell argument rules in `scripts/pack_cli.py`, shared by both packs.

Edit code/configuration/docs locally. Run data generation, numerical tests,
training, prediction, and plotting on SSH host `haiqin`. Use the frozen uv
environment; keep `scripts/base_pack/data_processing/matlab_reference/` static and do
not execute MATLAB. Follow [remote validation](doc/workflows/remote-validation.md)
for environment setup and command scope. From the local project root:

```bash
./scripts/run.sh base_pack validate --suite smoke
```

The dispatcher uses only the standard library. Snapshot uncommitted/new source
for validation; never overwrite haiqin's existing checkout or original inputs.
Reject silent CPU fallback in the validation workflow. If CUDA fails only
inside a sandbox, recheck outside it before changing drivers/packages.

## Architecture Contracts

- Code roots are `scripts/<pack>/data_processing/`,
  `scripts/<pack>/localization_networks/`, and `scripts/<pack>/visualization/`.
  Use canonical `scripts.<pack>.localization_networks.*` imports within the
  selected pack; do not add compatibility aliases for the removed roots.
  Shared network logic belongs in each pack's `common/`; method-specific layouts
  stay in their packages. Reuse `common/h5_utils.py` for paths and splits.
- Method packages are `rbd/`, `elm/`, `scm/`, and standalone `grnn/`, each with
  `main.py`. Public shell method names and output directories remain uppercase.
  Use `MethodBindings` for shared neural CLI/train/predict dependencies.
- RBD `/X` is `[sample,element,frequency,real_imag]`. ELM/SCM use
  `[sample,pair,frequency,real_imag]`; ELM pairs require `i<j`, SCM `i<=j`.
  Pair indices are stored 1-based in column-major order.
- Each method owns `network/model.py:MODEL_REGISTRY` and `network/models/`.
  RBD config uses `input_elements` plus `input_freq_bins`; ELM/SCM use
  `input_pairs` plus `input_freq_bins`. Models return normalized `[batch,1]`.
  Keep GRNN outside trainable model registries.
- Joint inputs concatenate canonical RBD, ELM, SCM blocks along feature rows;
  do not label the concatenated axis as elements or pairs. Preserve existing
  neural layers and channels. `raw_concat` and training-fitted `block_rms` are
  distinct experimental conditions; prediction must reuse fitted scales.
  Joint generation requires aligned Ns=1 samples, physical axes, and sources.
- Keep pack, feature combination, and fusion identity in run/artifact metadata.
  Never silently reuse a checkpoint/reference or fetch a run for another pack.
- Preserve `DatasetBundle`: `train_dataset`, `val_dataset`, `train_idx`,
  `val_idx`, `train_labels`, `input_shape`, `split_mode`, `source_paths`.
  Its `input_signature` compares frequency, element/depth, and pair axes.
  Current checkpoints/references require this contract; reject missing fields.
- Generation configuration is defined in `generation_config.py`; reject
  unknown/invalid fields. Adaptive frequencies fit valid training windows only.
- Timestamped network exports use `MMDD_HHMMSS`. Resolve latest matching
  artifacts by mtime through `common/paths.py:latest_time_suffixed_path`.
- Update `pyproject.toml` and `uv.lock` together. Local delivery is
  `output/data/<run_id>/` for logs/tables/reports and `output/figure/<run_id>/`
  for figures; remote intermediate artifacts use the run's `outputs/`.

## Git and Required Reading

The Git whitelist tracks README/AGENTS, doc, scripts, simulation, and uv project
files. Never commit `Origindata/`, `outputs/`, `output/`, local environments,
macOS junk (`._*`, `.DS_Store`), or stray files. Commit only on user request;
write detailed Chinese commit messages. Push only when requested.

Before changing sensitive areas, read:

- [HDF5 datasets and splits](doc/datasets/hdf5-datasets-and-splits.md):
  generation, windows, frequency selection, and stored fields.
- [Python extension guide](doc/development/python-extension-guide.md):
  loaders, model registration, and interface constraints.
- [Workflow guides](doc/README.md#workflows): execution and result contracts.
- [Project progress](doc/overview/project-progress.md): verified scope and gaps.
- [Experiment packs](doc/packs/README.md): pack ownership, commands, and joint
  feature comparison contracts.

Keep the language of the document being edited. Check current source and
reports before stating behavior or scientific validation claims.
