function [ratio_freq, freq_hz, feature_info] = ELM_extract_ratio_feature( ...
    signal_time_seg, fs, denom_floor_relative, freq_bin_idx, pair_i, pair_j, ...
    snapshot_num_samples, snapshot_step_samples)
%ELM_EXTRACT_RATIO_FEATURE Build pairwise element frequency-ratio features.
%
% For each one-sided frequency bin, this returns
%   ratio_freq(i,j,f) = sum_s X_i,s(f) * conj(X_j,s(f)) ...
%       / (sum_s abs(X_j,s(f))^2 + floor).
%
% Inputs:
%   signal_time_seg        N x Nt time-domain array segment
%   fs                     sampling frequency [Hz]
%   denom_floor_relative   relative denominator power floor
%   freq_bin_idx           optional one-sided FFT bin indices to keep
%   pair_i                 numerator element indices for vector output
%   pair_j                 denominator element indices for vector output
%   snapshot_num_samples   optional snapshot FFT length in samples
%   snapshot_step_samples  optional step between snapshots in samples
%
% Outputs:
%   ratio_freq             P x F pair-vector tensor
%   freq_hz                1 x F one-sided frequency vector [Hz]
%   feature_info           struct with numerical metadata

if nargin < 3 || isempty(denom_floor_relative)
    denom_floor_relative = 1e-6;
end

if ~ismatrix(signal_time_seg)
    error('signal_time_seg must be a 2-D matrix with size N x Nt.');
end

[num_elements, segment_num_samples_total] = size(signal_time_seg);
if num_elements < 1 || segment_num_samples_total < 2
    error('signal_time_seg must contain at least one element and two samples.');
end

if ~isscalar(fs) || fs <= 0
    error('fs must be a positive scalar sampling frequency.');
end

if ~isscalar(denom_floor_relative) || ~isnumeric(denom_floor_relative) || ...
        ~isreal(denom_floor_relative) || denom_floor_relative < 0
    error('denom_floor_relative must be a non-negative real scalar.');
end

if nargin < 7 || isempty(snapshot_num_samples)
    snapshot_num_samples = segment_num_samples_total;
end
if nargin < 8 || isempty(snapshot_step_samples)
    snapshot_step_samples = snapshot_num_samples;
end
if ~isscalar(snapshot_num_samples) || snapshot_num_samples < 2 || ...
        snapshot_num_samples ~= round(snapshot_num_samples)
    error('snapshot_num_samples must be an integer >= 2.');
end
if ~isscalar(snapshot_step_samples) || snapshot_step_samples < 1 || ...
        snapshot_step_samples ~= round(snapshot_step_samples)
    error('snapshot_step_samples must be a positive integer.');
end
if snapshot_num_samples > segment_num_samples_total
    error('snapshot_num_samples cannot exceed segment length.');
end

snapshot_start_idx = ...
    1:snapshot_step_samples:(segment_num_samples_total - snapshot_num_samples + 1);
num_snapshots = numel(snapshot_start_idx);

full_num_freq_bins = floor(snapshot_num_samples / 2) + 1;
full_freq_hz = (0:full_num_freq_bins - 1) * fs / snapshot_num_samples;

if nargin < 4 || isempty(freq_bin_idx)
    freq_bin_idx = 1:full_num_freq_bins;
end

freq_bin_idx = double(freq_bin_idx(:).');
if any(~isfinite(freq_bin_idx)) || any(freq_bin_idx ~= round(freq_bin_idx)) || ...
        any(freq_bin_idx < 1) || any(freq_bin_idx > full_num_freq_bins)
    error('freq_bin_idx must contain valid one-sided FFT bin indices.');
end

freq_hz = full_freq_hz(freq_bin_idx);
num_freq_bins = numel(freq_hz);

if nargin < 6 || isempty(pair_i) || isempty(pair_j)
    error('pair_i and pair_j are required; ELM stores only pair-vector features.');
end

pair_i = double(pair_i(:));
pair_j = double(pair_j(:));
if numel(pair_i) ~= numel(pair_j)
    error('pair_i and pair_j must have the same number of entries.');
end
if any(~isfinite(pair_i)) || any(~isfinite(pair_j)) || ...
        any(pair_i ~= round(pair_i)) || any(pair_j ~= round(pair_j)) || ...
        any(pair_i < 1) || any(pair_i > num_elements) || ...
        any(pair_j < 1) || any(pair_j > num_elements)
    error('pair_i and pair_j must contain valid element indices.');
end

snapshot_freq = complex(zeros(num_elements, num_freq_bins, num_snapshots));
for snapshot_idx = 1:num_snapshots
    start_idx = snapshot_start_idx(snapshot_idx);
    stop_idx = start_idx + snapshot_num_samples - 1;
    snapshot_time = signal_time_seg(:, start_idx:stop_idx);
    signal_freq = fft(snapshot_time, snapshot_num_samples, 2);
    signal_freq = signal_freq(:, 1:full_num_freq_bins);
    snapshot_freq(:, :, snapshot_idx) = signal_freq(:, freq_bin_idx);
end

signal_power = abs(snapshot_freq).^2;
finite_power = signal_power(isfinite(signal_power));
if isempty(finite_power)
    median_power = 0;
else
    median_power = median(finite_power);
end
denom_floor = denom_floor_relative * median_power;

numer = snapshot_freq(pair_i, :, :);
denom = snapshot_freq(pair_j, :, :);
ratio_freq = sum(numer .* conj(denom), 3) ./ ...
    (sum(abs(denom).^2, 3) + denom_floor + eps);

feature_info = struct();
feature_info.mode = 'element_pairwise_frequency_ratio';
feature_info.definition = ...
    'ratio_freq(i,j,f)=sum_s X_i,s(f)*conj(X_j,s(f))/(sum_s |X_j,s(f)|^2+floor)';
feature_info.num_elements = num_elements;
feature_info.output_layout = 'pair_vector';
feature_info.num_pairs = numel(pair_i);
feature_info.pair_numerator_element_idx = uint16(pair_i);
feature_info.pair_denominator_element_idx = uint16(pair_j);
feature_info.segment_num_samples = segment_num_samples_total;
feature_info.snapshot_num_samples = snapshot_num_samples;
feature_info.snapshot_step_samples = snapshot_step_samples;
feature_info.num_snapshots = num_snapshots;
feature_info.full_num_freq_bins = full_num_freq_bins;
feature_info.num_freq_bins = num_freq_bins;
feature_info.selected_fft_bin_idx = uint32(freq_bin_idx);
feature_info.denom_floor_relative = denom_floor_relative;
feature_info.denom_floor = denom_floor;
end
