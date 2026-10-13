# Base Pack

`scripts/base_pack/` preserves the original independent RBD, ELM, and SCM
experiment. It contains the former `scripts/data_processing/`,
`scripts/localization_networks/`, and `scripts/visualization/` trees. The public
shell remains `scripts/run.sh`; Python imports and direct entry paths now
include `base_pack`.

```bash
./scripts/run.sh base_pack
./scripts/run.sh base_pack ELM complex_CNN
./scripts/run.sh base_pack SCM GRNN
./scripts/run.sh base_pack validate --suite smoke
```

The pack-only command runs all 15 models using the selected configuration.
Add `--dry-run` to inspect the complete plan or `--detach` to launch remotely
and return its run ID.

| Scope | Contract |
|---|---|
| Features | Independent RBD element features, ELM strict-upper pairs, SCM pairs including the diagonal |
| Neural models | Four families per route: complex CNN, real CNN, ResNet18, ResNet50 |
| GRNN | One build/predict route per method, with independent sigma search |
| Smoke plan | Three complex CNNs and three GRNNs |
| Experiment plan | 12 neural models and three GRNNs |
| Fusion | None; `--fusion` and `[fusion]` are invalid |

The migration changes package locations and explicit pack selection. It does
not redefine the features, split rules, network structures, or baseline
hyperparameters. Existing artifact compatibility still depends on the actual
input and checkpoint/reference contract; moving the source does not make
incompatible artifacts valid.

The [HDF5 contract](../datasets/hdf5-datasets-and-splits.md),
[model architecture](../methods/trainable-model-architecture.md), and
[GRNN guide](../workflows/standalone-grnn.md) remain the detailed references.
Use [command defaults](../workflows/command-defaults.md) for the original
settings and [remote workflow](../workflows/remote-validation.md) for options.

Historical run IDs and metrics remain in
[project progress](../overview/project-progress.md). Treat those as evidence
from their recorded source snapshots; pack migration requires its own checks.
