# Joint Feature Pack Training and Scaling

`scripts/network_training/joint_feature_pack/` trains the four baseline neural families (`joint/main.py`) and standalone GRNN (`grnn/main.py`) on single or concatenated RBD, ELM, and SCM blocks. Block concatenation order (`RBD → ELM → SCM`), row counts (`21..462`), and `Ns=1` alignment rules are defined in [joint feature concatenation](../data_processing/joint-features.md); shell flags (`--fusion`) live in [remote validation](../remote_workflow/remote-validation.md).

## Fusion Modes and Fitted Block Scales

`raw_concat` and `block_rms` are distinct experimental conditions persisted in every checkpoint, GRNN reference, and run report:

- **`raw_concat`**: Feeds concatenated raw block values directly (subsequent per-model input normalization in CNN/ResNet still applies).
- **`block_rms`** (default): Divides complex entries in block $m$ by a training-fitted scalar preserving within-block phase:

```text
s_m = sqrt(mean(real(X_m)^2 + imag(X_m)^2))
X_joint = concat_rows(X_m / s_m for each selected block m)
```

Fitting and reuse rules:
- **Neural models**: Fit $s_m$ strictly on the optimizer-training subset after validation splitting; validation, test prediction, and resume reuse the saved scales without refitting. Checkpoints also verify ordered training/validation file paths, byte sizes, and SHA256 hashes on resume.
- **GRNN**: Fit $s_m$ independently inside each CV training fold during sigma selection, then refit on all training rows for the final reference.

## Architecture and Boundary Notes

Neural channel widths, convolution kernels, and regression heads stay identical to [baseline trainable architectures](trainable-model-architecture.md), using `input_features` for the concatenated row height (`[batch, 2, rows, F]`); adaptive spatial pooling absorbs taller inputs. Note that equal per-entry RMS does not equalize total block energy because row counts ($21, 210, 231$) differ, and 2D convolutions cross adjacent block boundaries. Current experimental comparisons are summarized in [project progress](../project-progress.md).
