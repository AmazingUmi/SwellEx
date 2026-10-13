# Scripts Organization

Use `./scripts/run.sh PACK` from the local root for the full experiment on haiqin,
or add `METHOD ARCH` for one model. `scripts/pack_cli.py` owns shared option
parsing and implicit command selection; the shell only locates the project and
executes Python through uv.
The [command guide](../workflows/remote-validation.md) explains execution;
this page identifies the source to change. Pack ownership and joint-specific
contracts are in the [pack index](../packs/README.md).

## Directory Map

| Directory | Responsibility | Entry point |
|---|---|---|
| `scripts/` | Select an explicit pack; no legacy imports at the removed code roots | `run.sh` |
| `scripts/base_pack/data_processing/` | Configuration, raw inputs, feature math, windows/splits, and HDF5 generation | `generate.py`, `validate.py` |
| `scripts/base_pack/localization_networks/rbd/` | RBD loader, registry, and model configuration | `main.py train/predict` |
| `scripts/base_pack/localization_networks/elm/` | ELM loader, registry, and model configuration | `main.py train/predict` |
| `scripts/base_pack/localization_networks/scm/` | SCM loader, registry, and model configuration | `main.py train/predict` |
| `scripts/base_pack/localization_networks/grnn/` | RBD/ELM/SCM reference samples, sigma selection, and shared kernel prediction | `main.py build/predict --method METHOD` |
| `scripts/base_pack/localization_networks/common/` | Shared neural CLI, training/prediction, checkpoints, physical input signatures, and paths | Imported by method packages |
| `scripts/base_pack/localization_networks/workflow/` | Shell dispatch, remote execution/cache/retrieval, explicit local CPU GRNN, and existing-result comparison | `remote.py`, `local_grnn.py`, `compare_results.py`; `worker.py` is internal |
| `scripts/base_pack/localization_networks/validation/` | Runtime/model checks, dataset checks, fixtures, and structured reports | `validate.py quick/datasets` |
| `scripts/base_pack/visualization/` | Result discovery, consistency checks, aggregate figures, and provenance | `plot_results.py` |
| `scripts/joint_feature_pack/` | Separate generation, fusion loaders/contracts, existing neural/GRNN families, workflow, and plots | The selected pack's `localization_networks/workflow/remote.py` |

## Naming and Shared Logic

- Python packages and modules use lowercase snake_case; method packages are
  `rbd`, `elm`, `scm`, and `grnn`. Old numbered and `_method` paths were removed.
- Every method uses `main.py`. Model files are `complex_cnn.py`, `real_cnn.py`,
  and `resnet.py`; public model registry names remain method-specific.
- Import through `scripts.base_pack.localization_networks.*`. Entry scripts add the
  repository root once before importing; internal modules use package imports.
  Joint modules use `scripts.joint_feature_pack.*`; do not import or mutate the
  baseline implementation to add joint-only behavior.
- Neural wrappers provide `MethodBindings` to shared code. Tensor layout,
  registry, and model configuration remain owned by each method.
- `validation/validate.py` parses commands; `quick.py` checks runtime/models,
  `datasets.py` checks generated data, and `reporting.py` writes reports.
- `grnn/` owns reference and kernel algorithms; `workflow/local_grnn.py` owns
  generation, execution and delivery. `workflow/compare_results.py` reads saved
  evidence using only the standard library and never launches a model.

Public shell names remain `RBD/ELM/SCM` and `complex_CNN/real_CNN/ResNet18/ResNet50`
plus `GRNN` for each method in `base_pack`. Its complete plan has 15 combinations; smoke has
three complex CNNs and three GRNNs. Output roots retain uppercase method names. Use the
[extension guide](python-extension-guide.md) when adding a model or loader.
The joint pack adds canonical feature combinations and fusion identity; its
complete plan contains 35 models for one fusion mode.

## Validation and Change Boundaries

Edit source locally and run numerical tests through the frozen remote workflow.
Focused tests live beside the code in `tests/`; the quick suite includes neural
contracts, GRNN, generation identity, workflow, and visualization tests.

Keep `base_pack/data_processing/matlab_reference/` static. Do not execute MATLAB or commit
datasets, model artifacts, generated figures, or local environments.
