function frequency_tag = DS_make_frequency_selection_tag( ...
    frequency_selection_modes, frequency_config)
%DS_MAKE_FREQUENCY_SELECTION_TAG Build a stable tag for selected frequencies.

if nargin < 2 || isempty(frequency_config)
    frequency_config = struct();
end
if ~isfield(frequency_config, 'mel_num_bins') || ...
        isempty(frequency_config.mel_num_bins)
    frequency_config.mel_num_bins = 64;
end
if ~isfield(frequency_config, 'adapt_num_bins') || ...
        isempty(frequency_config.adapt_num_bins)
    frequency_config.adapt_num_bins = 16;
end

frequency_selection_modes = string(frequency_selection_modes(:).');
normalized_modes = lower(strtrim(frequency_selection_modes));
if any(normalized_modes == "full") && numel(normalized_modes) > 1
    error('frequency_selection_modes cannot combine "full" with other modes.');
end
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
frequency_tag = DS_sanitize_dataset_variant_tag(frequency_tag);
end
