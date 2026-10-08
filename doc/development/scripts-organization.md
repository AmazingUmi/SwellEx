# Scripts Organization

Use `./scripts/run.sh METHOD ARCH` from the local root to run a model on haiqin.
The [command guide](../workflows/remote-validation.md) explains execution;
this page identifies the source to change.

## Directory Map

| Directory | Responsibility | Entry point |
|---|---|---|
| `scripts/data_processing/` | Configuration, raw inputs, feature math, windows/splits, and HDF5 generation | `generate.py`, `validate.py` |
| `scripts/localization_networks/rbd/` | RBD loader, registry, and model configuration | `main.py train/predict` |
| `scripts/localization_networks/elm/` | ELM loader, registry, and model configuration | `main.py train/predict` |
| `scripts/localization_networks/scm/` | SCM loader, registry, and model configuration | `main.py train/predict` |
| `scripts/localization_networks/grnn/` | SCM reference samples, sigma selection, and kernel prediction | `main.py build/predict` |
| `scripts/localization_networks/common/` | Shared neural CLI, training/prediction, checkpoints, physical input signatures, and paths | Imported by method packages |
| `scripts/localization_networks/workflow/` | Local SSH dispatch, remote execution, cache, process timeouts, and result retrieval | `remote.py`; `worker.py` is internal |
| `scripts/localization_networks/validation/` | Runtime/model checks, dataset checks, fixtures, and structured reports | `validate.py quick/datasets` |
| `scripts/visualization/` | Result discovery, consistency checks, aggregate figures, and provenance | `plot_results.py` |

## Naming and Shared Logic

- Python packages and modules use lowercase snake_case; method packages are
  `rbd`, `elm`, `scm`, and `grnn`. Old numbered and `_method` paths were removed.
- Every method uses `main.py`. Model files are `complex_cnn.py`, `real_cnn.py`,
  and `resnet.py`; public model registry names remain method-specific.
- Import through `scripts.localization_networks.*`. Entry scripts add the
  repository root once before importing; internal modules use package imports.
- Neural wrappers provide `MethodBindings` to shared code. Tensor layout,
  registry, and model configuration remain owned by each method.
- `validation/validate.py` parses commands; `quick.py` checks runtime/models,
  `datasets.py` checks generated data, and `reporting.py` writes reports.

Public shell names remain `RBD/ELM/SCM` and `complex_CNN/real_CNN/ResNet18/ResNet50`
plus `SCM GRNN`. Output roots also retain uppercase method names. Use the
[extension guide](python-extension-guide.md) when adding a model or loader.

## Validation and Change Boundaries

Edit source locally and run numerical tests through the frozen remote workflow.
Focused tests live beside the code in `tests/`; the quick suite includes neural
contracts, GRNN, generation identity, workflow, and visualization tests.

Keep `data_processing/matlab_reference/` static. Do not execute MATLAB or commit
datasets, model artifacts, generated figures, or local environments.
