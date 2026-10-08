function dataset_variant_tag = RBD_make_dataset_variant_tag( ...
    frequency_selection_modes, frequency_config, segment_duration_s, ...
    segment_step_s, normalize_spectrum, use_plane_wave, rbd_beam_selection, ...
    rbd_frequency_estimation)
%RBD_MAKE_DATASET_VARIANT_TAG Build a stable RBD dataset variant tag.
% Example:
%   rbd_green_mel64_estfull_seg1s_step1s_norm0_pw0_bestbeam

if nargin < 8 || isempty(rbd_frequency_estimation)
    rbd_frequency_estimation = "full";
end

frequency_tag = DS_make_frequency_selection_tag( ...
    frequency_selection_modes, frequency_config);
segment_tag = DS_format_seconds_for_tag(segment_duration_s);
step_tag = DS_format_seconds_for_tag(segment_step_s);
estimation_tag = lower(strtrim(string(rbd_frequency_estimation)));
switch estimation_tag
    case "full"
        estimation_tag = "estfull";
    case "selected"
        estimation_tag = "estsel";
    otherwise
        error('Unsupported RBD frequency estimation mode: %s.', ...
            rbd_frequency_estimation);
end

beam_selection = lower(strtrim(convertCharsToStrings(rbd_beam_selection)));
switch beam_selection
    case "best"
        beam_tag = "bestbeam";
    case "multipath"
        beam_tag = "multipath";
    otherwise
        error('Unsupported RBD beam selection mode: %s.', rbd_beam_selection);
end

dataset_variant_tag = sprintf( ...
    'rbd_green_%s_%s_seg%ss_step%ss_norm%d_pw%d_%s', ...
    frequency_tag, estimation_tag, segment_tag, step_tag, ...
    logical(normalize_spectrum), logical(use_plane_wave), char(beam_tag));
dataset_variant_tag = DS_sanitize_dataset_variant_tag(dataset_variant_tag);
end
