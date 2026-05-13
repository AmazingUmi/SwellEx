function [freq_bin_idx, selected_freq_hz, mel_center_freq_hz] = ...
    SCM_make_mel_frequency_bins(full_freq_hz, num_mel_bins, min_freq_hz, max_freq_hz)
%SCM_MAKE_MEL_FREQUENCY_BINS Select FFT bins nearest to Mel-spaced centers.
if ~isvector(full_freq_hz) || isempty(full_freq_hz)
    error('full_freq_hz must be a non-empty frequency vector.');
end

full_freq_hz = double(full_freq_hz(:).');
if any(~isfinite(full_freq_hz)) || any(diff(full_freq_hz) <= 0)
    error('full_freq_hz must be finite and strictly increasing.');
end

if ~isscalar(num_mel_bins) || ~isnumeric(num_mel_bins) || ...
        ~isfinite(num_mel_bins) || num_mel_bins < 1
    error('num_mel_bins must be a positive scalar.');
end
num_mel_bins = round(num_mel_bins);

available_min_hz = full_freq_hz(1);
available_max_hz = full_freq_hz(end);

if nargin < 3 || isempty(min_freq_hz)
    min_freq_hz = available_min_hz;
end
if nargin < 4 || isempty(max_freq_hz)
    max_freq_hz = available_max_hz;
end

min_freq_hz = max(double(min_freq_hz), available_min_hz);
max_freq_hz = min(double(max_freq_hz), available_max_hz);
if ~isfinite(min_freq_hz) || ~isfinite(max_freq_hz) || min_freq_hz >= max_freq_hz
    error('Frequency selection range must satisfy min_freq_hz < max_freq_hz.');
end

min_mel = hz_to_mel(min_freq_hz);
max_mel = hz_to_mel(max_freq_hz);
mel_center_freq_hz = mel_to_hz(linspace(min_mel, max_mel, num_mel_bins));

freq_bin_idx = zeros(1, num_mel_bins);
for center_idx = 1:num_mel_bins
    [~, freq_bin_idx(center_idx)] = min(abs(full_freq_hz - ...
        mel_center_freq_hz(center_idx)));
end

[freq_bin_idx, unique_pos] = unique(freq_bin_idx, 'stable');
mel_center_freq_hz = mel_center_freq_hz(unique_pos);
selected_freq_hz = full_freq_hz(freq_bin_idx);
end

function mel = hz_to_mel(freq_hz)
mel = 2595 * log10(1 + freq_hz / 700);
end

function freq_hz = mel_to_hz(mel)
freq_hz = 700 * (10 .^ (mel / 2595) - 1);
end
