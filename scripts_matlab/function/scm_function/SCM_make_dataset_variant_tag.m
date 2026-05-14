function dataset_variant_tag = SCM_make_dataset_variant_tag( ...
    frequency_selection_modes, frequency_config, segment_duration_s, ...
    num_snapshots_per_segment, snapshot_overlap_count)
%SCM_MAKE_DATASET_VARIANT_TAG Build a stable SCM dataset variant tag.
% Example:
%   scm_upper_diag_mel64_snap1s_ns4_ov3

frequency_selection_modes = string(frequency_selection_modes(:).');
frequency_parts = strings(1, numel(frequency_selection_modes));

for mode_idx = 1:numel(frequency_selection_modes)
    mode_name = lower(strtrim(frequency_selection_modes(mode_idx)));
    switch mode_name
        case "full"
            frequency_parts(mode_idx) = "full";
        case "mel"
            frequency_parts(mode_idx) = sprintf('mel%d', ...
                round(frequency_config.mel_num_bins));
        case "deep"
            frequency_parts(mode_idx) = "deep";
        case "shallow"
            frequency_parts(mode_idx) = "shallow";
        case "adapt"
            frequency_parts(mode_idx) = sprintf('adapt%d', ...
                round(frequency_config.adapt_num_bins));
        otherwise
            error('Unsupported frequency selection mode: %s.', mode_name);
    end
end

frequency_tag = strjoin(frequency_parts, '_');
snapshot_tag = format_seconds_for_tag(segment_duration_s);

dataset_variant_tag = sprintf( ...
    'scm_upper_diag_%s_snap%ss_ns%d_ov%d', ...
    frequency_tag, snapshot_tag, ...
    round(num_snapshots_per_segment), round(snapshot_overlap_count));
dataset_variant_tag = DS_sanitize_dataset_variant_tag(dataset_variant_tag);
end

function tag = format_seconds_for_tag(value_s)
if abs(value_s - round(value_s)) < 100 * eps(max(1, abs(value_s)))
    tag = sprintf('%d', round(value_s));
else
    tag = regexprep(sprintf('%.3g', value_s), '\.', 'p');
end
end
