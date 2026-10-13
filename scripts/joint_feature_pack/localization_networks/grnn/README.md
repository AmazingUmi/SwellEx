# Standalone Joint-Feature GRNN

Use `scripts/joint_feature_pack/localization_networks/grnn/main.py build --method METHOD`
to save reference samples and `predict --method METHOD` to apply Gaussian-kernel
regression. METHOD is one of the seven feature selections in the
[joint pack contract](../../../../doc/packs/joint-feature-pack.md).
This package stays outside trainable CNN/ResNet registries and has no epochs or optimizer.

Pass the generated joint dataset directory explicitly with `--data`; the remote
workflow creates these directories under `outputs/JointDatasets/`. References and predictions default to
`outputs/networks_results/<METHOD>/<method>_grnn_range/`. Input method and axes must match the saved
reference signature and saved fusion/scaling contract; references missing the
current contract fields must be rebuilt. Block RMS is fitted within each CV
training fold and on all training rows for the final reference.

Use `./scripts/run.sh joint_feature_pack --method METHOD --network GRNN --fusion block_rms` for
the remote build/predict workflow. See the
[method note](../../../../doc/methods/grnn.md) for the shared kernel formula
and evaluation limits; the joint pack contract defines its preprocessing.
