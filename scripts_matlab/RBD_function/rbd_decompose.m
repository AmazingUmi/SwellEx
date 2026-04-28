function [green_freq, freq_hz, result] = rbd_decompose( ...
    signal_time_seg, fs, theta_vec, tau_matrix, varargin)
%RBD_DECOMPOSE Estimate element-wise frequency-domain Green's functions.
%
% This function contains the core RBD steps used in RBD_main.m:
%   1. FFT of a fixed-length multi-element signal segment
%   2. Frequency-domain normalization
%   3. Bartlett beamforming over steering angles
%   4. Best-angle selection
%   5. Equivalent Green's function estimation by phase rotation
%
% Inputs:
%   signal_time_seg       N x Nt time-domain array signal segment
%   fs                    sampling frequency [Hz]
%   theta_vec             1 x Ntheta steering angles [rad]
%   tau_matrix            N x Ntheta precomputed delay matrix [s]
%
% Name-value options:
%   'NormalizeSpectrum'   true to normalize frequency data (default true)
%
% Outputs:
%   green_freq            N x Nf one-sided frequency-domain Green's function
%   freq_hz               1 x Nf one-sided frequency vector [Hz]
%   result                struct with intermediate RBD products and metadata

normalize_spectrum = parse_options(varargin{:});

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

theta_vec = theta_vec(:).';
if isempty(theta_vec)
    error('theta_vec must contain at least one steering angle.');
end

if ~isequal(size(tau_matrix), [num_elements, numel(theta_vec)])
    error('tau_matrix must have size N x Ntheta. Expected %d x %d.', ...
        num_elements, numel(theta_vec));
end

freq_hz_full = (0:segment_num_samples - 1) * fs / segment_num_samples;
num_freq_bins = floor(segment_num_samples / 2) + 1;
freq_hz = freq_hz_full(1:num_freq_bins);
omega = 2 * pi * freq_hz;

signal_freq_seg = fft(signal_time_seg, segment_num_samples, 2);
signal_freq_seg = signal_freq_seg(:, 1:num_freq_bins);

signal_freq_scale = 1;
if normalize_spectrum
    signal_freq_power_by_element = sum(abs(signal_freq_seg).^2, 2);
    signal_freq_scale = sqrt(sum(signal_freq_power_by_element) / num_elements);
    signal_freq_seg = signal_freq_seg ./ (signal_freq_scale + eps);
end

beam_output = bartlett_beamformer(signal_freq_seg, omega, ...
    tau_matrix, num_elements);

beam_power = sum(abs(beam_output).^2, 1);
beam_power(~isfinite(beam_power)) = -Inf;
[~, best_angle_idx] = max(beam_power);
theta_best = theta_vec(best_angle_idx);

beam_best = beam_output(:, best_angle_idx);
phase_rotation = exp(-1j * angle(beam_best)).';
green_freq = signal_freq_seg .* phase_rotation;

result = struct();
result.green_freq = green_freq;
result.freq_hz = freq_hz;
result.omega = omega;
result.signal_freq_seg = signal_freq_seg;
result.signal_freq_scale = signal_freq_scale;
result.tau_matrix = tau_matrix;
result.beam_output = beam_output;
result.beam_power = beam_power;
result.best_angle_idx = best_angle_idx;
result.theta_best = theta_best;
result.phase_rotation = phase_rotation;
result.theta_vec = theta_vec;
result.normalize_spectrum = normalize_spectrum;
result.num_elements = num_elements;
result.segment_num_samples = segment_num_samples;
result.num_freq_bins = num_freq_bins;
end

function normalize_spectrum = parse_options(varargin)
normalize_spectrum = true;

if mod(numel(varargin), 2) ~= 0
    error('Options must be name-value pairs.');
end

for option_idx = 1:2:numel(varargin)
    option_name = lower(string(varargin{option_idx}));
    option_value = varargin{option_idx + 1};

    switch option_name
        case "normalizespectrum"
            normalize_spectrum = logical(option_value);
        otherwise
            error('Unknown option: %s.', varargin{option_idx});
    end
end
end
