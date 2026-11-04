function [scm_feature, freq_hz, feature_info] = SCM_extract_feature( ...
    signal_time_seg, fs, snapshot_num_samples, snapshot_step_samples, ...
    freq_bin_idx, pair_i, pair_j, norm_floor)
%SCM_EXTRACT_FEATURE Build normalized spatial covariance matrix features.
%
% Outputs:
%   scm_feature            P x F pair-vector tensor, upper triangle i <= j
%   freq_hz                1 x F one-sided frequency vector [Hz]
%   feature_info           struct with numerical metadata

if nargin < 8 || isempty(norm_floor)
    norm_floor = 1.0e-12;
end

if ~ismatrix(signal_time_seg)
    error('signal_time_seg must be a 2-D matrix with size M x Nt.');
end

[num_elements, segment_num_samples] = size(signal_time_seg);
if num_elements < 1 || segment_num_samples < 2
    error('signal_time_seg must contain at least one element and two samples.');
end

if ~isscalar(fs) || fs <= 0
    error('fs must be a positive scalar sampling frequency.');
end

if isempty(snapshot_num_samples)
    snapshot_num_samples = segment_num_samples;
end
if isempty(snapshot_step_samples)
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
if snapshot_num_samples > segment_num_samples
    error('snapshot_num_samples cannot exceed segment length.');
end

full_num_freq_bins = floor(snapshot_num_samples / 2) + 1;
full_freq_hz = (0:full_num_freq_bins - 1) * fs / snapshot_num_samples;

if nargin < 5 || isempty(freq_bin_idx)
    freq_bin_idx = 1:full_num_freq_bins;
end
freq_bin_idx = double(freq_bin_idx(:).');
if any(~isfinite(freq_bin_idx)) || any(freq_bin_idx ~= round(freq_bin_idx)) || ...
        any(freq_bin_idx < 1) || any(freq_bin_idx > full_num_freq_bins)
    error('freq_bin_idx must contain valid one-sided FFT bin indices.');
end

if nargin < 7 || isempty(pair_i) || isempty(pair_j)
    error('pair_i and pair_j are required for SCM pair-vector output.');
end
pair_i = double(pair_i(:));
pair_j = double(pair_j(:));
if numel(pair_i) ~= numel(pair_j)
    error('pair_i and pair_j must have the same number of entries.');
end
if any(~isfinite(pair_i)) || any(~isfinite(pair_j)) || ...
        any(pair_i ~= round(pair_i)) || any(pair_j ~= round(pair_j)) || ...
        any(pair_i < 1) || any(pair_i > num_elements) || ...
        any(pair_j < 1) || any(pair_j > num_elements) || any(pair_i > pair_j)
    error('pair_i and pair_j must contain valid upper-triangle element indices.');
end

snapshot_start_idx = 1:snapshot_step_samples:(segment_num_samples - snapshot_num_samples + 1);
num_snapshots = numel(snapshot_start_idx);
num_freq_bins = numel(freq_bin_idx);
num_pairs = numel(pair_i);
scm_feature = complex(zeros(num_pairs, num_freq_bins));

for freq_idx = 1:num_freq_bins
    bin_idx = freq_bin_idx(freq_idx);
    snapshot_vectors = complex(zeros(num_elements, num_snapshots));
    for snapshot_idx = 1:num_snapshots
        start_idx = snapshot_start_idx(snapshot_idx);
        stop_idx = start_idx + snapshot_num_samples - 1;
        snapshot_time = signal_time_seg(:, start_idx:stop_idx);
        snapshot_freq = fft(snapshot_time, snapshot_num_samples, 2);
        x = snapshot_freq(:, bin_idx);
        x_norm = x ./ max(norm(x), norm_floor);
        snapshot_vectors(:, snapshot_idx) = x_norm;
    end

    c_q = (snapshot_vectors * snapshot_vectors') / num_snapshots;
    scm_feature(:, freq_idx) = c_q(sub2ind([num_elements, num_elements], pair_i, pair_j));
end

freq_hz = full_freq_hz(freq_bin_idx);

feature_info = struct();
feature_info.mode = 'spatial_covariance_matrix';
feature_info.definition = 'C_q=mean_s((x_s/||x_s||)(x_s/||x_s||)^H)';
feature_info.output_layout = 'upper_triangle_pair_vector_with_diagonal';
feature_info.num_elements = num_elements;
feature_info.num_pairs = num_pairs;
feature_info.pair_numerator_element_idx = uint16(pair_i);
feature_info.pair_denominator_element_idx = uint16(pair_j);
feature_info.segment_num_samples = segment_num_samples;
feature_info.snapshot_num_samples = snapshot_num_samples;
feature_info.snapshot_step_samples = snapshot_step_samples;
feature_info.num_snapshots = num_snapshots;
feature_info.full_num_freq_bins = full_num_freq_bins;
feature_info.num_freq_bins = num_freq_bins;
feature_info.selected_fft_bin_idx = uint32(freq_bin_idx);
feature_info.norm_floor = norm_floor;
end
