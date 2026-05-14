function [freq_bin_idx, selected_freq_hz, selection_info] = ...
    DS_select_frequency_bins(full_freq_hz, frequency_selection_modes, config)
%DS_SELECT_FREQUENCY_BINS Select one-sided FFT bins for dataset features.
%
% Supported modes:
%   "full"    all one-sided FFT bins
%   "mel"     nearest bins to Mel-spaced center frequencies
%   "deep"    named deep-source target frequencies
%   "shallow" named shallow-source target frequencies
%   "adapt"   strongest dataset-level frequency bins by signal power

if nargin < 2 || isempty(frequency_selection_modes)
    frequency_selection_modes = "mel";
end
if nargin < 3 || isempty(config)
    config = struct();
end

if ~isvector(full_freq_hz) || isempty(full_freq_hz)
    error('full_freq_hz must be a non-empty frequency vector.');
end
full_freq_hz = double(full_freq_hz(:).');
if any(~isfinite(full_freq_hz)) || any(diff(full_freq_hz) <= 0)
    error('full_freq_hz must be finite and strictly increasing.');
end

frequency_selection_modes = string(frequency_selection_modes(:).');
if any(strlength(frequency_selection_modes) == 0)
    error('frequency_selection_modes must not contain empty mode names.');
end

default_deep_freq_hz = [49 64 79 94 112 130 148 166 201 235 283 338 388];
default_shallow_freq_hz = [109 127 145 163 198 232 280 335 385];

if ~isfield(config, 'mel_num_bins') || isempty(config.mel_num_bins)
    config.mel_num_bins = 64;
end
if ~isfield(config, 'mel_min_freq_hz') || isempty(config.mel_min_freq_hz)
    config.mel_min_freq_hz = full_freq_hz(1);
end
if ~isfield(config, 'mel_max_freq_hz') || isempty(config.mel_max_freq_hz)
    config.mel_max_freq_hz = full_freq_hz(end);
end
if ~isfield(config, 'deep_target_freq_hz') || isempty(config.deep_target_freq_hz)
    config.deep_target_freq_hz = default_deep_freq_hz;
end
if ~isfield(config, 'shallow_target_freq_hz') || isempty(config.shallow_target_freq_hz)
    config.shallow_target_freq_hz = default_shallow_freq_hz;
end
if ~isfield(config, 'adapt_num_bins') || isempty(config.adapt_num_bins)
    config.adapt_num_bins = 16;
end
if ~isfield(config, 'adapt_min_freq_hz') || isempty(config.adapt_min_freq_hz)
    config.adapt_min_freq_hz = max(1, full_freq_hz(1));
end
if ~isfield(config, 'adapt_max_freq_hz') || isempty(config.adapt_max_freq_hz)
    config.adapt_max_freq_hz = full_freq_hz(end);
end
if ~isfield(config, 'adapt_pooling') || isempty(config.adapt_pooling)
    config.adapt_pooling = 'mean_power_over_elements_snapshots_windows';
end

selected_bins = [];
mode_info = repmat(struct( ...
    'mode', '', ...
    'requested_freq_hz', [], ...
    'selected_fft_bin_idx', [], ...
    'selected_freq_hz', [], ...
    'mel_center_freq_hz', [], ...
    'adapt_power', []), 1, 0);

for mode_idx = 1:numel(frequency_selection_modes)
    mode_name = lower(strtrim(frequency_selection_modes(mode_idx)));
    info = struct();
    info.mode = char(mode_name);
    info.requested_freq_hz = [];
    info.selected_fft_bin_idx = [];
    info.selected_freq_hz = [];
    info.mel_center_freq_hz = [];
    info.adapt_power = [];

    switch mode_name
        case "full"
            mode_bins = 1:numel(full_freq_hz);

        case "mel"
            [mode_bins, ~, mel_center_freq_hz] = select_mel_bins( ...
                full_freq_hz, config.mel_num_bins, ...
                config.mel_min_freq_hz, config.mel_max_freq_hz);
            info.mel_center_freq_hz = mel_center_freq_hz;
            info.requested_freq_hz = mel_center_freq_hz;

        case "deep"
            requested_freq_hz = double(config.deep_target_freq_hz(:).');
            mode_bins = nearest_frequency_bins(full_freq_hz, requested_freq_hz);
            info.requested_freq_hz = requested_freq_hz;

        case "shallow"
            requested_freq_hz = double(config.shallow_target_freq_hz(:).');
            mode_bins = nearest_frequency_bins(full_freq_hz, requested_freq_hz);
            info.requested_freq_hz = requested_freq_hz;

        case "adapt"
            [mode_bins, adapt_power] = select_adaptive_bins(full_freq_hz, config);
            info.adapt_power = adapt_power;

        otherwise
            error('Unsupported frequency selection mode: %s.', mode_name);
    end

    mode_bins = unique(mode_bins, 'stable');
    info.selected_fft_bin_idx = uint32(mode_bins);
    info.selected_freq_hz = full_freq_hz(mode_bins);
    selected_bins = [selected_bins, mode_bins]; %#ok<AGROW>
    mode_info(end + 1) = info; %#ok<AGROW>
end

[freq_bin_idx, unique_pos] = unique(selected_bins);
selected_freq_hz = full_freq_hz(freq_bin_idx);

selection_info = struct();
selection_info.frequency_selection_modes = frequency_selection_modes;
selection_info.selected_fft_bin_idx = uint32(freq_bin_idx);
selection_info.selected_freq_hz = selected_freq_hz;
selection_info.num_freq_bins = numel(freq_bin_idx);
selection_info.mode_info = mode_info;
selection_info.has_duplicate_requested_bins = numel(unique_pos) < numel(selected_bins);
selection_info.config = remove_adaptive_data_fields(config);
end

function [freq_bin_idx, selected_freq_hz, mel_center_freq_hz] = ...
    select_mel_bins(full_freq_hz, num_mel_bins, min_freq_hz, max_freq_hz)
if ~isscalar(num_mel_bins) || ~isnumeric(num_mel_bins) || ...
        ~isfinite(num_mel_bins) || num_mel_bins < 1
    error('mel_num_bins must be a positive scalar.');
end
num_mel_bins = round(num_mel_bins);

min_freq_hz = max(double(min_freq_hz), full_freq_hz(1));
max_freq_hz = min(double(max_freq_hz), full_freq_hz(end));
if ~isfinite(min_freq_hz) || ~isfinite(max_freq_hz) || min_freq_hz >= max_freq_hz
    error('Mel frequency range must satisfy mel_min_freq_hz < mel_max_freq_hz.');
end

min_mel = hz_to_mel(min_freq_hz);
max_mel = hz_to_mel(max_freq_hz);
mel_center_freq_hz = mel_to_hz(linspace(min_mel, max_mel, num_mel_bins));
freq_bin_idx = nearest_frequency_bins(full_freq_hz, mel_center_freq_hz);
[freq_bin_idx, unique_pos] = unique(freq_bin_idx, 'stable');
mel_center_freq_hz = mel_center_freq_hz(unique_pos);
selected_freq_hz = full_freq_hz(freq_bin_idx);
end

function freq_bin_idx = nearest_frequency_bins(full_freq_hz, target_freq_hz)
if ~isvector(target_freq_hz) || isempty(target_freq_hz)
    error('Target frequency list must be a non-empty vector.');
end
target_freq_hz = double(target_freq_hz(:).');
if any(~isfinite(target_freq_hz))
    error('Target frequency list must contain finite values.');
end

freq_bin_idx = zeros(1, numel(target_freq_hz));
for target_idx = 1:numel(target_freq_hz)
    [~, freq_bin_idx(target_idx)] = min(abs(full_freq_hz - target_freq_hz(target_idx)));
end
end

function [freq_bin_idx, mean_power] = select_adaptive_bins(full_freq_hz, config)
required_fields = {'signal_time_full', 'segment_sample_start_idx', ...
    'segment_sample_stop_idx', 'fs', 'snapshot_num_samples', ...
    'snapshot_step_samples'};
for field_idx = 1:numel(required_fields)
    if ~isfield(config, required_fields{field_idx}) || ...
            isempty(config.(required_fields{field_idx}))
        error('Adaptive frequency selection requires config.%s.', ...
            required_fields{field_idx});
    end
end

signal_time_full = config.signal_time_full;
segment_sample_start_idx = config.segment_sample_start_idx;
segment_sample_stop_idx = config.segment_sample_stop_idx;
snapshot_num_samples = config.snapshot_num_samples;
snapshot_step_samples = config.snapshot_step_samples;

if isfield(config, 'adapt_candidate_segment_idx') && ...
        ~isempty(config.adapt_candidate_segment_idx)
    candidate_segment_idx = double(config.adapt_candidate_segment_idx(:).');
else
    candidate_segment_idx = 1:numel(segment_sample_start_idx);
end
candidate_segment_idx = candidate_segment_idx(isfinite(candidate_segment_idx) & ...
    candidate_segment_idx == round(candidate_segment_idx) & ...
    candidate_segment_idx >= 1 & ...
    candidate_segment_idx <= numel(segment_sample_start_idx));
candidate_segment_idx = unique(candidate_segment_idx, 'stable');
if isempty(candidate_segment_idx)
    error('Adaptive frequency selection has no valid candidate segments.');
end

if isfield(config, 'adapt_max_num_segments') && ...
        ~isempty(config.adapt_max_num_segments) && ...
        numel(candidate_segment_idx) > config.adapt_max_num_segments
    keep_idx = round(linspace(1, numel(candidate_segment_idx), ...
        config.adapt_max_num_segments));
    candidate_segment_idx = candidate_segment_idx(keep_idx);
end

freq_mask = full_freq_hz >= config.adapt_min_freq_hz & ...
    full_freq_hz <= config.adapt_max_freq_hz;
freq_mask(1) = freq_mask(1) && full_freq_hz(1) > 0;
candidate_freq_idx = find(freq_mask);
if isempty(candidate_freq_idx)
    error('Adaptive frequency selection range contains no FFT bins.');
end

full_num_freq_bins = numel(full_freq_hz);
power_sum = zeros(1, full_num_freq_bins);
power_count = 0;

for segment_pos = 1:numel(candidate_segment_idx)
    segment_idx = candidate_segment_idx(segment_pos);
    start_idx = segment_sample_start_idx(segment_idx);
    stop_idx = segment_sample_stop_idx(segment_idx);
    signal_time_seg = signal_time_full(:, start_idx:stop_idx);
    segment_num_samples_total = size(signal_time_seg, 2);
    snapshot_start_idx = ...
        1:snapshot_step_samples:(segment_num_samples_total - snapshot_num_samples + 1);

    for snapshot_idx = 1:numel(snapshot_start_idx)
        snapshot_start = snapshot_start_idx(snapshot_idx);
        snapshot_stop = snapshot_start + snapshot_num_samples - 1;
        snapshot_time = signal_time_seg(:, snapshot_start:snapshot_stop);
        signal_freq = fft(snapshot_time, snapshot_num_samples, 2);
        signal_freq = signal_freq(:, 1:full_num_freq_bins);
        snapshot_power = mean(abs(signal_freq).^2, 1, 'omitnan');
        power_sum = power_sum + snapshot_power;
        power_count = power_count + 1;
    end
end

if power_count < 1
    error('Adaptive frequency selection found no snapshots to analyze.');
end
mean_power = power_sum / power_count;
candidate_power = mean_power(candidate_freq_idx);
[~, order_idx] = sort(candidate_power, 'descend');
num_adapt_bins = min(round(config.adapt_num_bins), numel(order_idx));
freq_bin_idx = candidate_freq_idx(order_idx(1:num_adapt_bins));
freq_bin_idx = sort(freq_bin_idx);
end

function mel = hz_to_mel(freq_hz)
mel = 2595 * log10(1 + freq_hz / 700);
end

function freq_hz = mel_to_hz(mel)
freq_hz = 700 * (10 .^ (mel / 2595) - 1);
end

function config_out = remove_adaptive_data_fields(config_in)
config_out = config_in;
large_fields = {'signal_time_full', 'segment_sample_start_idx', ...
    'segment_sample_stop_idx', 'adapt_candidate_segment_idx'};
for field_idx = 1:numel(large_fields)
    if isfield(config_out, large_fields{field_idx})
        config_out = rmfield(config_out, large_fields{field_idx});
    end
end
end
