# Joint Feature Concatenation and Alignment

`scripts/data_processing/common/{joint_builder.py,joint_contract.py}` and `<pack>/{contract.py,joint_features.py}` build and validate concatenated multi-route complex features for `joint_feature_pack`, `MoE_data_scale` (`RBD_SCM`), and `MoE_learn` (`RBD_SCM`).

## Canonical Block Order and Row Counts

Selected single-route HDF5 blocks (`[S, rows_m, F, 2]`) concatenate along the row axis in fixed `RBD → ELM → SCM` order (`[batch, 2, rows, F]` in loaders; e.g. `[batch, 2, 252, 22]` for `RBD_SCM` and `[batch, 2, 462, 22]` for `RBD_ELM_SCM` at $E=21, F=22$):

| Selection | Concatenated blocks | Feature rows ($E=21$) | Packs |
|---|---|---:|---|
| `RBD` / `ELM` / `SCM` | Single canonical route | 21 / 210 / 231 | `joint_feature_pack`, `MoE_data_scale` |
| `RBD_SCM` | `RBD`, `SCM` | 252 | `joint_feature_pack`, `MoE_data_scale`, `MoE_learn` |
| `RBD_ELM` / `ELM_SCM` | Two blocks in `RBD → ELM → SCM` order | 231 / 441 | `joint_feature_pack` |
| `RBD_ELM_SCM` | `RBD`, `ELM`, `SCM` | 462 | `joint_feature_pack` |

Block boundaries and per-block physical signatures remain explicit in metadata; the concatenated axis is always labeled as **feature rows**, never elements or pairs.

## Alignment Contract

Joint generation and loading require `num_snapshots_per_segment = 1`, `snapshot_overlap_count = 0`, and disjoint base windows (`segment_step_s >= segment_duration_s`). Before concatenation, builders and loaders verify exact agreement across all participating routes and reject any mismatch in:

- Raw source provenance and event metadata
- Sample counts, `source_segment_idx` window identities, and `valid_sample` masks
- `y_range_km` labels and window `start` / `center` / `stop` timestamps
- `/frequency/freq_hz` bins and `/array/{element_idx,depth_m}` coordinates

Training-time block scaling (`raw_concat` vs training-fitted `block_rms`) belongs to `network_training`; see [joint feature pack](../network_training/joint-feature-pack.md), [MoE data scale](../network_training/moe-data-scale.md), and [continual MoE](../network_training/moe-learn.md).
