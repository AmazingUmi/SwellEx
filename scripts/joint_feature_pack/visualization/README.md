# Result Visualization

This package renders one recorded `joint_feature_pack` run with a canonical
model plan and explicit `raw_concat` or `block_rms` identity. The report,
figure context, metrics CSV, and rendering manifest retain pack and fusion.
Results from `base_pack` or an older plan are rejected.

Prediction panels use seven feature rows and common architecture columns;
ECDF panels separate architectures and compare at most seven feature inputs
per panel. All recorded models are validated before any display selection.

See the [joint pack guide](../../../doc/packs/joint-feature-pack.md) for input
and comparison semantics. The [visualization reference](../../../doc/workflows/result-visualization.md)
describes the shared figure definitions; direct rendering uses this pack's
`plot_results.py` on haiqin with the frozen environment.
