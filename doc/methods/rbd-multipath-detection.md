# RBD Multipath Peak Detection

`scripts/base_pack/data_processing/feature_math.py:rbd_decompose` forms equivalent Green
features from Bartlett beams. `rbd.beam_selection="best"` uses the strongest
angle; `"multipath"` uses accepted local peaks. The stored `/X` contains the
aggregated Green feature, so this selection affects network input directly.

## Configuration

```json
{"rbd": {"beam_selection": "multipath", "multipath_options": {
  "peak_threshold_db": -6, "min_separation_deg": 2,
  "max_num_peaks": 181, "sidelobe_reject_db": 3
}}}
```

`generation_config.py` validates these fields. The relative threshold limits
candidates; separation limits nearby accepted angles; the count caps accepted
peaks; the sidelobe margin controls rejection of predicted leakage.

## Selection and Aggregation

`select_beam_peaks` works on linear beam power, summed over frequencies:

1. Find local maxima with power at least
   `maximum_power * 10**(peak_threshold_db/10)`.
2. Visit candidates by descending power and enforce `min_separation_deg`
   against accepted angles.
3. Estimate leakage at each candidate from accepted beams' powers and the
   normalized steering-vector point-spread response. Reject a candidate when
   predicted leakage is positive and its power is at most
   `predicted_leakage * 10**(sidelobe_reject_db/10)`.
4. Stop at `max_num_peaks`; use the strongest angle if none survives.

Each accepted beam supplies a phase rotation of the array spectrum, normalized
per frequency by its array-vector norm plus float64 machine epsilon. This
gives components and their aggregate:

```text
component_k(:,f) = normalized_spectrum(:,f) * exp(-j*angle(beam(f,k)))
green_freq = sum_k weight_k * component_k
```

Individual non-finite or negative selected powers are replaced by zero.
If their sum is positive, weights are those powers divided by the sum;
otherwise weights are uniform over the accepted angles. The scan currently
has 181 angles from −90° to 90°. HDF5 angles are in radians; configured
separation is in degrees.

## HDF5 Diagnostics

`generation_core.py` writes the following under `/rbd` (`S` samples, `A` scan
angles). These fields are diagnostics, not extra model-input channels.

| Field | Shape | Meaning |
|---|---|---|
| `theta_vec_rad` | `[A,1]` | Scan grid |
| `theta_best_rad` | `[S,1]` | Strongest angle |
| `num_selected_angles` | `[S,1]` | Accepted count |
| `theta_selected_rad` | `[S,A]` | Accepted angles |
| `selected_beam_power` | `[S,A]` | Accepted linear powers |
| `beam_power` | `[S,A]` | Full linear scan powers |
| `signal_freq_scale` | `[S,1]` | Optional spectrum normalization scale |

Accepted slots are stored by descending power; unused slots are NaN.
The [HDF5 guide](../datasets/hdf5-datasets-and-splits.md) defines finite `/X`,
axis order, frequency modes, and Green-normalization checks.
