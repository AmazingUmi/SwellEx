# Joint Feature Pack

`scripts/joint_feature_pack/` evaluates whether combining RBD, ELM, and SCM
inputs improves range regression with the existing network architectures.
It starts from the base Python implementation and owns its experimental
loaders, contracts, configuration, workflow, and result handling.

## Select an Experiment

```bash
# Complete 35-model plan with the bundled configuration and block_rms
./scripts/run.sh joint_feature_pack

# Three feature blocks, each scaled using training samples
./scripts/run.sh joint_feature_pack RBD_ELM_SCM complex_CNN --fusion block_rms

# Two feature blocks without block scaling
./scripts/run.sh joint_feature_pack RBD_SCM ResNet18 --fusion raw_concat

# Same input comparison with a standalone GRNN
./scripts/run.sh joint_feature_pack RBD_ELM_SCM GRNN --fusion block_rms
```

| Selection | Blocks, in fixed order | Rows for 21 elements |
|---|---|---:|
| `RBD` | RBD | 21 |
| `ELM` | ELM | 210 |
| `SCM` | SCM | 231 |
| `RBD_ELM` | RBD, ELM | 231 |
| `RBD_SCM` | RBD, SCM | 252 |
| `ELM_SCM` | ELM, SCM | 441 |
| `RBD_ELM_SCM` | RBD, ELM, SCM | 462 |

Each selection supports `complex_CNN`, `real_CNN`, `ResNet18`, `ResNet50`, and
`GRNN`. The complete plan contains 28 neural models and seven GRNNs for one
fusion mode, split, and seed. `raw_concat` and `block_rms` are separate runs;
the complete plan does not implicitly evaluate both modes.
Use `./scripts/run.sh joint_feature_pack --fusion raw_concat` to run the full
plan under the other fusion mode, or add `--dry-run` to inspect either plan.

`validate --suite smoke` runs one complex CNN and one GRNN for each selection
(14 models); `validate --suite experiment` runs all 35. Use a copied joint
preset with a short valid interval and reduced epochs for engineering checks;
the default preset keeps the full interval and 100 neural epochs.

The default fusion mode is `block_rms`. Set `--fusion raw_concat|block_rms`,
or change `mode` in the selected joint preset's `[fusion]` table; the CLI
overrides the configuration. Unknown selections and fusion modes are rejected.
GRNN uses build/predict, so `--epochs` remains invalid for it.

## Representation and Fitted Scales

The input keeps real/imaginary channels and the common frequency axis. Selected
blocks concatenate along feature rows in canonical `RBD → ELM → SCM` order.
For element count E and frequency count F, the block heights are E,
E(E−1)/2, and E(E+1)/2. The full three-way tensor is
`[batch, 2, E(E+1), F]`; at E=21 and F=22 it is `[batch,2,462,22]`.
Block boundaries and each block's physical signature remain explicit: the
combined row axis is not a single element axis or a single pair axis.

`raw_concat` preserves block values. `block_rms` divides every complex value
in block m by a positive training-fitted scalar:

```text
s_m = sqrt(mean(real(X_m)^2 + imag(X_m)^2))
X_joint = concat_rows(X_m / s_m for each selected block m)
```

The mean spans the fitting samples, block rows, and frequencies. Real and
imaginary values use the same scalar, retaining block phase relationships.
Neural fitting uses the actual optimizer-training subset after its validation
split; validation and test inputs reuse that fitted scale. GRNN cross-validation
must fit scales within each fold's training reference, and the final reference
fits on all training rows. Scales and fusion identity must accompany saved
artifacts and be reused for prediction; they must not be refitted on held-out
inputs.

Neural checkpoints also record the ordered training/validation source paths,
file sizes, and complete SHA256 hashes. Resume checks this identity for both
the resumed checkpoint and its historical best checkpoint, rejecting changed
values or reordered rows. Prediction validates the saved contract without
requiring test data to have the training files' hashes.

Each model's existing subsequent preprocessing remains part of the experiment.
The real CNN uses magnitudes and per-sample magnitude normalization; complex
CNN and ResNet retain real/imaginary inputs and their existing channel
normalization. `raw_concat` therefore means no added block scaling, not no
preprocessing anywhere in the model.

Network layers, channel counts, and regression heads stay unchanged; adaptive
pooling accepts the changed input height. Joint inputs increase computation
and may increase memory use. Equal block RMS does not equalize total block
energy because the row counts differ. Convolutions cross block boundaries;
fixed block order is part of this initial baseline.

## Alignment Contract

The first implementation requires `num_snapshots_per_segment = 1`, zero
snapshot overlap, and disjoint base windows. RBD uses base windows while ELM
and SCM can otherwise combine multiple snapshots, so equal row counts alone
are insufficient for fusion.

Before concatenation, require matching source identities, split/sample
identities, valid masks, range labels, start/center/stop times, frequency axes,
element indices, and depths. Retain each block's own pair axis where present.
Missing metadata or any disagreement must reject the combination rather than
silently zip, trim, or reorder samples. Training/validation/test partitions
must use identical sample identities across compared selections.

The supported Python model entry points are `localization_networks/joint/main.py`
and `localization_networks/grnn/main.py`. Copied native `rbd/elm/scm` data/model
helpers are internal dependencies; their old executable train/predict entries
are absent from this pack.

## Comparison and Validation

First run a short remote pipeline check covering contracts, all four neural
families, GRNN, saved-state prediction, and result delivery. Then compare the
seven selections under both fusion modes using fixed test rows, training
settings, and at least three seeds. Report MAE, RMSE, tail errors, runtime, and
the exact input/fusion/split identity. Keep `full_interleaved` and the two
directions of `range_overlap` separate.

The seven selections test whether the three-way input improves on the best
single route and whether a pair is sufficient. Single-route controls inside
this pack use the same fusion and preprocessing rules as its joint selections;
historical base metrics are not automatically matched controls.

Figures and `metrics.csv` retain the pack and fusion identity. Prediction
grids use one row per feature combination and common architecture columns;
error ECDFs use separate architecture panels with at most seven input curves.
The renderer validates the complete recorded suite before applying a display
selection, and rejects base-pack reports or incomplete joint plans.

At single-snapshot geometry the three representations derive from the same
signals and can be algebraically related. A gain would show that the model
benefits from these input representations under the tested settings; it would
not establish added independent physical information or cross-track
generalization. Current verification status belongs in
[project progress](../overview/project-progress.md), not in this design contract.
