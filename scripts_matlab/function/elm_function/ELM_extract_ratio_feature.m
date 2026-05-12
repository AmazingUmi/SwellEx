function [ratio_freq, freq_hz, feature_info] = ELM_extract_ratio_feature( ...
    signal_time_seg, fs, denom_floor_relative, freq_bin_idx)
%ELM_EXTRACT_RATIO_FEATURE Build pairwise element frequency-ratio features.
%
% For each one-sided frequency bin, this returns
%   ratio_freq(i,j,f) = X_i(f) / X_j(f)
% using the stable equivalent
%   X_i(f) * conj(X_j(f)) / (abs(X_j(f))^2 + floor).
%
% Inputs:
%   signal_time_seg        N x Nt time-domain array segment
%   fs                     sampling frequency [Hz]
%   denom_floor_relative   relative denominator power floor
%   freq_bin_idx           optional one-sided FFT bin indices to keep
%
% Outputs:
%   ratio_freq             N x N x F complex pairwise ratio tensor
%   freq_hz                1 x F one-sided frequency vector [Hz]
%   feature_info           struct with numerical metadata

if nargin < 3 || isempty(denom_floor_relative)
    denom_floor_relative = 1e-6;
end

if ~ismatrix(signal_time_seg)
    error('signal_time_seg must be a 2-D matrix with size N x Nt.');
end

[num_elements, segment_num_samples] = size(signal_time_seg);
if num_elements < 1 || segment_num_samples < 2
    error('signal_time_seg must contain at least one element and two samples.');
end

if ~isscalar(fs) || fs <= 0
    error('fs must be a positive scalar sampling frequency.');
end

if ~isscalar(denom_floor_relative) || ~isnumeric(denom_floor_relative) || ...
        ~isreal(denom_floor_relative) || denom_floor_relative < 0
    error('denom_floor_relative must be a non-negative real scalar.');
end

full_num_freq_bins = floor(segment_num_samples / 2) + 1;
full_freq_hz = (0:full_num_freq_bins - 1) * fs / segment_num_samples;

if nargin < 4 || isempty(freq_bin_idx)
    freq_bin_idx = 1:full_num_freq_bins;
end

freq_bin_idx = double(freq_bin_idx(:).');
if any(~isfinite(freq_bin_idx)) || any(freq_bin_idx ~= round(freq_bin_idx)) || ...
        any(freq_bin_idx < 1) || any(freq_bin_idx > full_num_freq_bins)
    error('freq_bin_idx must contain valid one-sided FFT bin indices.');
end

signal_freq = fft(signal_time_seg, segment_num_samples, 2);
signal_freq = signal_freq(:, 1:full_num_freq_bins);
signal_freq = signal_freq(:, freq_bin_idx);
freq_hz = full_freq_hz(freq_bin_idx);
num_freq_bins = numel(freq_hz);

signal_power = abs(signal_freq).^2;
finite_power = signal_power(isfinite(signal_power));
if isempty(finite_power)
    median_power = 0;
else
    median_power = median(finite_power);
end
denom_floor = denom_floor_relative * median_power;

numer = reshape(signal_freq, num_elements, 1, num_freq_bins);
denom = reshape(signal_freq, 1, num_elements, num_freq_bins);
ratio_freq = numer .* conj(denom) ./ (abs(denom).^2 + denom_floor + eps);

feature_info = struct();
feature_info.mode = 'element_pairwise_frequency_ratio';
feature_info.definition = 'ratio_freq(i,j,f)=FFT(element_i,f)/FFT(element_j,f)';
feature_info.num_elements = num_elements;
feature_info.segment_num_samples = segment_num_samples;
feature_info.full_num_freq_bins = full_num_freq_bins;
feature_info.num_freq_bins = num_freq_bins;
feature_info.selected_fft_bin_idx = uint32(freq_bin_idx);
feature_info.denom_floor_relative = denom_floor_relative;
feature_info.denom_floor = denom_floor;
end
