# RBD Multipath Peak Detection

## Purpose

The RBD preprocessing can now use multiple Bartlett beam-power peaks instead of
only the strongest steering angle. This is controlled by `multipath_beam` in:

- `scripts_matlab/RBD_main.m`
- `scripts_matlab/Signals_Analysis.m`
- `scripts_matlab/Signals_Segmentation.m`

When enabled, the selected arrivals are used to construct the equivalent
frequency-domain Green's function. Dataset files also store the selected angles
and beam diagnostics under the `/rbd` HDF5 group.

## Configuration

```matlab
multipath_beam = true;
if multipath_beam
    multipath_peak_threshold_db = -6;
    multipath_min_separation_deg = 2;
    multipath_max_num_peaks = Inf;
    multipath_sidelobe_reject_db = 3;
end
```

Parameter meanings:

- `multipath_beam`: use multiple accepted beam-power peaks when `true`; use
  only the strongest peak when `false`
- `multipath_peak_threshold_db`: relative beam-power threshold below the
  strongest peak
- `multipath_min_separation_deg`: minimum angular separation between accepted
  peaks
- `multipath_max_num_peaks`: maximum number of accepted peaks
- `multipath_sidelobe_reject_db`: margin used when rejecting a candidate that is
  explainable as sidelobe leakage from already accepted peaks

## Algorithm

`rbd_decompose` performs the following steps after Bartlett beamforming:

1. Compute `beam_power` and `beam_power_db` over `theta_vec`.
2. Find local maxima above `multipath_peak_threshold_db` relative to the
   strongest peak.
3. Sort candidate peaks by descending beam power.
4. Accept candidates that satisfy `multipath_min_separation_deg` from already
   accepted angles.
5. For non-first candidates, estimate sidelobe leakage from already accepted
   peaks using the steering-vector point-spread response. Reject the candidate
   when its power is not greater than the predicted leakage by
   `multipath_sidelobe_reject_db`.
6. Stop after `multipath_max_num_peaks` accepted peaks.
7. If no candidate survives, fall back to the strongest beam angle.

For each accepted angle, RBD computes a phase-rotated Green's-function
component. The final `green_freq` written to `/X` is the sum of all accepted
components:

```text
green_freq = sum(green_freq_components, arrival_axis)
```

## HDF5 Outputs

`Signals_Segmentation.m` writes these additional RBD fields into every split
HDF5 file:

```text
/rbd/theta_vec_rad          beam scan angle grid [num_angles]
/rbd/theta_best_rad         strongest beam angle per sample [num_samples, 1]
/rbd/num_selected_angles    accepted peak count per sample [num_samples, 1]
/rbd/theta_selected_rad     accepted angles [num_samples, num_arrival_slots]
/rbd/selected_beam_power    accepted beam powers [num_samples, num_arrival_slots]
/rbd/beam_power             full beam-power scan [num_samples, num_angles]
/rbd/signal_freq_scale      spectrum normalization scale [num_samples, 1]
```

`num_arrival_slots` currently equals `numel(theta_vec)`, so every possible
accepted peak can be stored. Rows in `theta_selected_rad` and
`selected_beam_power` are sorted by descending accepted beam power before
writing, and unused slots are filled with `NaN`.

## Notes For Python Training

The current Python range-regression datasets still read `/X`, `/y_range_km`,
`/valid_sample`, and optional timing/split fields. The multipath fields are
diagnostic metadata and do not change the model input tensor shape:

```text
x: [batch, 2, element, frequency]
```

Because `/X` already contains the summed multipath Green's function, training
uses the multipath-aware feature whenever the MATLAB dataset was generated with
`multipath_beam = true`.
