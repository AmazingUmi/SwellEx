function [green_freq, freq_hz, result] = rbd_decompose( ...
    signal_time_seg, fs, theta_vec, tau_matrix, varargin)
%RBD_DECOMPOSE Estimate element-wise frequency-domain Green's functions.
%
% This function contains the core RBD steps used in RBD_main.m:
%   1. FFT of a fixed-length multi-element signal segment
%   2. Frequency-domain normalization
%   3. Bartlett beamforming over steering angles
%   4. Best-angle or multi-peak angle selection
%   5. Equivalent Green's function estimation by phase rotation
%
% Inputs:
%   signal_time_seg       N x Nt time-domain array signal segment
%   fs                    sampling frequency [Hz]
%   theta_vec             1 x Ntheta steering angles [rad]
%   tau_matrix            N x Ntheta precomputed delay matrix [s]
%
% Name-value options:
%   'NormalizeSpectrum'            true to normalize frequency data
%                                  (default true)
%   'multipath_beam'               true to use all detected beam-power
%                                  peaks instead of only the strongest
%                                  angle (default false)
%   'MultipathPeakThresholdDb'     relative peak threshold in dB
%                                  (default -6)
%   'MultipathMinSeparationDeg'    minimum separation between selected
%                                  peaks in degrees (default 2)
%   'MultipathMaxNumPeaks'         maximum number of selected peaks
%                                  (default Inf)
%   'MultipathSidelobeRejectDb'    candidate peaks must exceed the
%                                  predicted sidelobe leakage from already
%                                  selected peaks by this margin in dB
%                                  (default 3)
%
% Outputs:
%   green_freq            N x Nf one-sided frequency-domain Green's function
%   freq_hz               1 x Nf one-sided frequency vector [Hz]
%   result                struct with intermediate RBD products and metadata

[normalize_spectrum, multipath_beam, multipath_peak_threshold_db, ...
    multipath_min_separation_deg, multipath_max_num_peaks, ...
    multipath_sidelobe_reject_db] = ...
    parse_options(varargin{:});

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

beam_power_db = compute_relative_power_db(beam_power);
if multipath_beam
    [selected_angle_idx, sidelobe_rejected_angle_idx, ...
        sidelobe_prediction_power, sidelobe_psf_power] = ...
        detect_beam_peaks(beam_power, theta_vec, tau_matrix, omega, ...
        multipath_peak_threshold_db, multipath_min_separation_deg, ...
        multipath_max_num_peaks, multipath_sidelobe_reject_db);
else
    selected_angle_idx = best_angle_idx;
    sidelobe_rejected_angle_idx = [];
    sidelobe_prediction_power = zeros(size(beam_power));
    sidelobe_psf_power = [];
end

num_selected_angles = numel(selected_angle_idx);
phase_rotation_components = zeros(num_selected_angles, num_freq_bins);
green_freq_components = zeros(num_elements, num_freq_bins, num_selected_angles);
green_freq_weights = compute_green_freq_weights( ...
    beam_power(selected_angle_idx), num_selected_angles);

for selected_idx = 1:num_selected_angles
    angle_idx = selected_angle_idx(selected_idx);
    beam_selected = beam_output(:, angle_idx);
    phase_rotation_components(selected_idx, :) = ...
        exp(-1j * angle(beam_selected)).';
    green_freq_components(:, :, selected_idx) = ...
        signal_freq_seg .* phase_rotation_components(selected_idx, :);
end

green_freq = sum(green_freq_components .* ...
    reshape(green_freq_weights, 1, 1, num_selected_angles), 3);
phase_rotation = phase_rotation_components(1, :);

result = struct();
result.green_freq = green_freq;
result.freq_hz = freq_hz;
result.omega = omega;
result.signal_freq_seg = signal_freq_seg;
result.signal_freq_scale = signal_freq_scale;
result.tau_matrix = tau_matrix;
result.beam_output = beam_output;
result.beam_power = beam_power;
result.beam_power_db = beam_power_db;
result.best_angle_idx = best_angle_idx;
result.theta_best = theta_best;
result.selected_angle_idx = selected_angle_idx;
result.theta_selected = theta_vec(selected_angle_idx);
result.selected_beam_power = beam_power(selected_angle_idx);
result.green_freq_weights = green_freq_weights;
result.sidelobe_rejected_angle_idx = sidelobe_rejected_angle_idx;
result.theta_sidelobe_rejected = theta_vec(sidelobe_rejected_angle_idx);
result.sidelobe_prediction_power = sidelobe_prediction_power;
result.sidelobe_psf_power = sidelobe_psf_power;
result.phase_rotation = phase_rotation;
result.phase_rotation_components = phase_rotation_components;
result.green_freq_components = green_freq_components;
result.num_selected_angles = num_selected_angles;
result.theta_vec = theta_vec;
result.normalize_spectrum = normalize_spectrum;
result.multipath_beam = multipath_beam;
result.multipath_peak_threshold_db = multipath_peak_threshold_db;
result.multipath_min_separation_deg = multipath_min_separation_deg;
result.multipath_max_num_peaks = multipath_max_num_peaks;
result.multipath_sidelobe_reject_db = multipath_sidelobe_reject_db;
result.num_elements = num_elements;
result.segment_num_samples = segment_num_samples;
result.num_freq_bins = num_freq_bins;
end

function [normalize_spectrum, multipath_beam, multipath_peak_threshold_db, ...
    multipath_min_separation_deg, multipath_max_num_peaks, ...
    multipath_sidelobe_reject_db] = parse_options(varargin)
normalize_spectrum = true;
multipath_beam = false;
multipath_peak_threshold_db = -6;
multipath_min_separation_deg = 2;
multipath_max_num_peaks = Inf;
multipath_sidelobe_reject_db = 3;

if mod(numel(varargin), 2) ~= 0
    error('Options must be name-value pairs.');
end

for option_idx = 1:2:numel(varargin)
    option_name = lower(string(varargin{option_idx}));
    option_value = varargin{option_idx + 1};

    switch option_name
        case "normalizespectrum"
            normalize_spectrum = logical(option_value);
        case {"multipath_beam", "multipathbeam"}
            multipath_beam = logical(option_value);
        case "multipathpeakthresholddb"
            multipath_peak_threshold_db = option_value;
        case "multipathminseparationdeg"
            multipath_min_separation_deg = option_value;
        case "multipathmaxnumpeaks"
            multipath_max_num_peaks = option_value;
        case "multipathsideloberejectdb"
            multipath_sidelobe_reject_db = option_value;
        otherwise
            error('Unknown option: %s.', varargin{option_idx});
    end
end

validate_scalar_logical(normalize_spectrum, 'NormalizeSpectrum');
validate_scalar_logical(multipath_beam, 'multipath_beam');
validate_real_scalar(multipath_peak_threshold_db, ...
    'MultipathPeakThresholdDb');
validate_real_scalar(multipath_min_separation_deg, ...
    'MultipathMinSeparationDeg');
validate_real_scalar(multipath_max_num_peaks, ...
    'MultipathMaxNumPeaks');
validate_real_scalar(multipath_sidelobe_reject_db, ...
    'MultipathSidelobeRejectDb');

if multipath_min_separation_deg < 0
    error('MultipathMinSeparationDeg must be nonnegative.');
end

if multipath_max_num_peaks < 1
    error('MultipathMaxNumPeaks must be at least 1.');
end
end

function beam_power_db = compute_relative_power_db(beam_power)
max_power = max(beam_power);
beam_power_db = -Inf(size(beam_power));

if ~isfinite(max_power)
    return;
end

relative_power = beam_power ./ (max_power + eps);
relative_power(~isfinite(relative_power) | relative_power < 0) = 0;
beam_power_db = 10 * log10(relative_power + eps);
end

function green_freq_weights = compute_green_freq_weights( ...
    selected_beam_power, num_selected_angles)
selected_beam_power = selected_beam_power(:).';
selected_beam_power(~isfinite(selected_beam_power) | selected_beam_power < 0) = 0;

total_power = sum(selected_beam_power);

if total_power > 0
    green_freq_weights = selected_beam_power ./ total_power;
else
    green_freq_weights = ones(1, num_selected_angles) ./ num_selected_angles;
end
end

function [selected_angle_idx, sidelobe_rejected_angle_idx, ...
    sidelobe_prediction_power, sidelobe_psf_power] = detect_beam_peaks( ...
    beam_power, theta_vec, tau_matrix, omega, threshold_db, ...
    min_separation_deg, max_num_peaks, sidelobe_reject_db)

num_angles = numel(beam_power);
[~, best_angle_idx] = max(beam_power);
max_power = beam_power(best_angle_idx);
sidelobe_rejected_angle_idx = [];
sidelobe_prediction_power = zeros(size(beam_power));
sidelobe_psf_power = compute_beam_psf_power(tau_matrix, omega);

if num_angles == 1 || ~isfinite(max_power)
    selected_angle_idx = best_angle_idx;
    return;
end

relative_threshold = 10^(threshold_db / 10);
power_threshold = max_power * relative_threshold;
candidate_idx = [];

for angle_idx = 1:num_angles
    power_value = beam_power(angle_idx);

    if ~isfinite(power_value) || power_value < power_threshold
        continue;
    end

    if is_local_peak(beam_power, angle_idx)
        candidate_idx(end + 1) = angle_idx; %#ok<AGROW>
    end
end

if isempty(candidate_idx)
    selected_angle_idx = best_angle_idx;
    return;
end

[~, sort_idx] = sort(beam_power(candidate_idx), 'descend');
candidate_idx = candidate_idx(sort_idx);
selected_angle_idx = [];
min_separation_rad = min_separation_deg * pi / 180;
sidelobe_reject_factor = 10^(sidelobe_reject_db / 10);

for candidate_pos = 1:numel(candidate_idx)
    angle_idx = candidate_idx(candidate_pos);

    if ~isempty(selected_angle_idx) && any(abs(theta_vec(angle_idx) - ...
            theta_vec(selected_angle_idx)) < min_separation_rad)
        continue;
    end

    if ~isempty(selected_angle_idx)
        predicted_power = sum(beam_power(selected_angle_idx) .* ...
            sidelobe_psf_power(angle_idx, selected_angle_idx));
        sidelobe_prediction_power(angle_idx) = predicted_power;

        if predicted_power > 0 && beam_power(angle_idx) <= ...
                predicted_power * sidelobe_reject_factor
            sidelobe_rejected_angle_idx(end + 1) = angle_idx; %#ok<AGROW>
            continue;
        end
    end

    selected_angle_idx(end + 1) = angle_idx; %#ok<AGROW>

    if numel(selected_angle_idx) >= max_num_peaks
        break;
    end
end

if isempty(selected_angle_idx)
    selected_angle_idx = best_angle_idx;
end

selected_angle_idx = sort(selected_angle_idx);
sidelobe_rejected_angle_idx = sort(sidelobe_rejected_angle_idx);
end

function sidelobe_psf_power = compute_beam_psf_power(tau_matrix, omega)
persistent cached_tau_matrix cached_omega cached_sidelobe_psf_power

if ~isempty(cached_sidelobe_psf_power) && isequaln(tau_matrix, cached_tau_matrix) && ...
        isequal(omega, cached_omega)
    sidelobe_psf_power = cached_sidelobe_psf_power;
    return;
end

num_angles = size(tau_matrix, 2);
sidelobe_psf_power = zeros(num_angles, num_angles);

for source_idx = 1:num_angles
    source_tau = tau_matrix(:, source_idx);

    for scan_idx = 1:num_angles
        scan_tau = tau_matrix(:, scan_idx);
        valid_idx = ~isnan(source_tau) & ~isnan(scan_tau);

        if ~any(valid_idx)
            sidelobe_psf_power(scan_idx, source_idx) = NaN;
            continue;
        end

        delay_difference = source_tau(valid_idx) - scan_tau(valid_idx);
        array_response = mean(exp(1j * delay_difference * omega), 1);
        sidelobe_psf_power(scan_idx, source_idx) = ...
            sum(abs(array_response).^2);
    end

    max_response = max(sidelobe_psf_power(:, source_idx));
    if isfinite(max_response) && max_response > 0
        sidelobe_psf_power(:, source_idx) = ...
            sidelobe_psf_power(:, source_idx) ./ max_response;
    end
end

sidelobe_psf_power(~isfinite(sidelobe_psf_power)) = 0;
cached_tau_matrix = tau_matrix;
cached_omega = omega;
cached_sidelobe_psf_power = sidelobe_psf_power;
end

function is_peak = is_local_peak(beam_power, angle_idx)
num_angles = numel(beam_power);
power_value = beam_power(angle_idx);

if num_angles == 1
    is_peak = true;
elseif angle_idx == 1
    is_peak = power_value > beam_power(2);
elseif angle_idx == num_angles
    is_peak = power_value > beam_power(num_angles - 1);
else
    is_peak = power_value >= beam_power(angle_idx - 1) && ...
        power_value > beam_power(angle_idx + 1);
end
end

function validate_scalar_logical(value, option_name)
if ~isscalar(value) || ~(islogical(value) || isnumeric(value))
    error('%s must be a scalar logical value.', option_name);
end
end

function validate_real_scalar(value, option_name)
if ~isscalar(value) || ~isnumeric(value) || ~isreal(value) || isnan(value)
    error('%s must be a real scalar value.', option_name);
end
end
