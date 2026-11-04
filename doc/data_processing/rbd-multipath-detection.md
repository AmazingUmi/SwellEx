# RBD Multipath Peak Detection

`feature_math.py:rbd_decompose` forms equivalent Green features from 181-angle (`-90°..90°`) Bartlett beamforming. `rbd.beam_selection="best"` uses the strongest angle; `"multipath"` aggregates accepted local peaks into `/X` (see [HDF5 datasets](hdf5-datasets-and-splits.md)).

## Standalone Generation JSON Options

Workflow TOML uses the defaults below; override `rbd.multipath_options` in standalone generation JSON:

| Parameter | Default | Constraint / meaning |
|---|---|---|
| `peak_threshold_db` | `-6` | Finite dB threshold relative to maximum frequency-summed beam power |
| `min_separation_deg` | `2` | Minimum angular separation (`>= 0°`) from already accepted peaks |
| `max_num_peaks` | `181` | Positive integer cap on accepted peaks |
| `sidelobe_reject_db` | `3` | Required dB margin above steering-vector point-spread leakage from accepted peaks |

## Selection, Aggregation, and `/rbd` Diagnostics

`select_beam_peaks` scans local maxima above `max_power * 10**(peak_threshold_db/10)` in descending power order, enforces `min_separation_deg` and `sidelobe_reject_db`, caps at `max_num_peaks`, and falls back to the strongest angle if empty. Each accepted angle $k$ phase-rotates the unit-norm array spectrum and combines by normalized linear beam power:

```text
component_k(:,f) = normalized_spectrum(:,f) * exp(-j * angle(beam(f,k)))
green_freq = sum_k weight_k * component_k
```

Diagnostic datasets saved under `/rbd` (not fed to models; unused slots are `NaN`): `theta_vec_rad` `[A,1]`, `theta_best_rad` `[S,1]`, `num_selected_angles` `[S,1]`, `theta_selected_rad` `[S,A]`, `selected_beam_power` `[S,A]`, `beam_power` `[S,A]`, and `signal_freq_scale` `[S,1]`.
