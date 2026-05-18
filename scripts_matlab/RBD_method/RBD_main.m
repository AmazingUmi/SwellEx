%% RBD (Ray-Based Deconvolution)
% Beamformer: Bartlett (conventional time-delay beamformer)
%
% This script:
%   1. Loads VLA geometry and the sound-speed profile
%   2. Loads all channel signals, centers them, then extracts one segment
%   3. Computes steering delays
%   4. Applies the Bartlett beamformer
%   5. Estimates the equivalent Green's function

%% Environment setup
clear; close all; clc;
try
    tmp = matlab.desktop.editor.getActive;
    script_dir = fileparts(tmp.Filename);
catch
    script_dir = fileparts(mfilename('fullpath'));
end
scripts_dir = fileparts(script_dir);
project_dir = fileparts(scripts_dir);
cd(script_dir);
addpath(script_dir);
addpath(genpath(fullfile(scripts_dir, 'function')));
origindata_dir = DS_get_origindata_dir(project_dir);
clear tmp;

%% User parameters
fs = 1500;                     % [Hz]
segment_duration_s = 1.0;      % segment duration [s]
segment_start_s = 400.0;       % segment start time [s]
segment_num_samples = round(fs * segment_duration_s);
segment_start_idx = round(fs * segment_start_s) + 1;

theta_vec = linspace(-90, 90, 181) * pi / 180;   % [rad]
use_plane_wave = false;

% RBD feature extraction
normalize_spectrum = true;
rbd_beam_selection = "multipath";   % "best" or "multipath"
rbd_multipath_options = struct();
rbd_multipath_options.peak_threshold_db = -6;
rbd_multipath_options.min_separation_deg = 2;
rbd_multipath_options.max_num_peaks = Inf;
rbd_multipath_options.sidelobe_reject_db = 3;

%% Array and environment
position_file = fullfile(origindata_dir, 'positions', 'positions_vla.txt');
position_table = readmatrix(position_file);   % [channel_index, depth_m]
array_depths_m = flip(position_table(:, 2).');
num_elements = numel(array_depths_m);

event_dir = fullfile(origindata_dir, 'events', 'S5');
ctd_data = load(fullfile(event_dir, "CTD_i9605.mat"));
sound_speed_depth_m = ctd_data.T.depth_m;
sound_speed_ms = ctd_data.T.sound_speed_ms;
[sound_speed_depth_m, sound_speed_ms] = RBD_extend_sound_speed_profile( ...
    sound_speed_depth_m, sound_speed_ms, max(array_depths_m));

clear position_file position_table ctd_data;

%% Load time-domain data
fprintf('Loading VLA time series...\n');

channel_data_dir = fullfile(event_dir, "vla_matfiles");

% Read all channel signals first so later we can reuse the centered
% full-length data for multiple time segments.
first_channel_file = sprintf('S5_VLA_NO_%d.mat', 1);
first_channel_data = load(fullfile(channel_data_dir, first_channel_file));
first_channel_signal = first_channel_data.x(:).';

signal_time_full = zeros(num_elements, numel(first_channel_signal));
signal_time_full(1, :) = first_channel_signal;

for channel_idx = 2:num_elements
    channel_file = sprintf('S5_VLA_NO_%d.mat', channel_idx);
    channel_data = load(fullfile(channel_data_dir, channel_file));
    channel_signal = channel_data.x(:).';

    if numel(channel_signal) ~= size(signal_time_full, 2)
        error('Channel %d sample count does not match channel 1.', channel_idx);
    end

    signal_time_full(channel_idx, :) = channel_signal;
end

% Center each channel after loading the complete data.
signal_time_full = signal_time_full - mean(signal_time_full, 2);

% Extract the selected segment from the centered full-length signals.
segment_stop_idx = segment_start_idx + segment_num_samples - 1;
signal_time_seg = signal_time_full(:, segment_start_idx:segment_stop_idx);

fprintf('Time segment ready.\n');
clear first_channel_data first_channel_signal first_channel_file;
clear channel_data channel_signal channel_idx channel_file channel_data_dir;
clear event_dir signal_time_full segment_stop_idx;

%% RBD decomposition
fprintf('Running RBD decomposition...\n');

tau_matrix = RBD_compute_tau(theta_vec, array_depths_m, ...
    sound_speed_ms, sound_speed_depth_m, use_plane_wave);

rbd_beam_selection = lower(strtrim(convertCharsToStrings(rbd_beam_selection)));
switch rbd_beam_selection
    case "best"
        multipath_beam = false;
    case "multipath"
        multipath_beam = true;
    otherwise
        error('Unsupported rbd_beam_selection: %s.', rbd_beam_selection);
end

rbd_options = {'NormalizeSpectrum', normalize_spectrum, ...
    'multipath_beam', multipath_beam};
if multipath_beam
    rbd_multipath_options = RBD_validate_multipath_options( ...
        rbd_multipath_options);
    rbd_options = [rbd_options, { ...
        'MultipathPeakThresholdDb', ...
        rbd_multipath_options.peak_threshold_db, ...
        'MultipathMinSeparationDeg', ...
        rbd_multipath_options.min_separation_deg, ...
        'MultipathMaxNumPeaks', ...
        rbd_multipath_options.max_num_peaks, ...
        'MultipathSidelobeRejectDb', ...
        rbd_multipath_options.sidelobe_reject_db}];
end

[green_freq, freq_hz, rbd_result] = RBD_decompose( ...
    signal_time_seg, fs, theta_vec, tau_matrix, rbd_options{:});

beam_power = rbd_result.beam_power;
theta_best = rbd_result.theta_best;
theta_plot_arrivals = theta_best;

fprintf('  Best steering angle: %.2f deg\n', theta_best * 180 / pi);
if multipath_beam
    [~, arrival_sort_idx] = sort(rbd_result.selected_beam_power, 'descend');
    theta_arrival = rbd_result.theta_selected(arrival_sort_idx);
    arrival_power_db = rbd_result.beam_power_db( ...
        rbd_result.selected_angle_idx(arrival_sort_idx));
    theta_plot_arrivals = theta_arrival;

    fprintf('  Multipath beam: enabled\n');
    fprintf('  Accepted steering angles (%d): %s deg\n', ...
        rbd_result.num_selected_angles, ...
        sprintf('%.2f ', theta_arrival * 180 / pi));
    fprintf('  Accepted relative powers: %s dB\n', ...
        sprintf('%.2f ', arrival_power_db));
    fprintf(['  Peak threshold = %.2f dB, min separation = %.2f deg, ', ...
        'sidelobe reject margin = %.2f dB\n'], ...
        rbd_multipath_options.peak_threshold_db, ...
        rbd_multipath_options.min_separation_deg, ...
        rbd_multipath_options.sidelobe_reject_db);

    if isempty(rbd_result.theta_sidelobe_rejected)
        fprintf('  Sidelobe-rejected candidate angles: none\n');
    else
        fprintf('  Sidelobe-rejected candidate angles (%d): %s deg\n', ...
            numel(rbd_result.theta_sidelobe_rejected), ...
            sprintf('%.2f ', rbd_result.theta_sidelobe_rejected * 180 / pi));
    end
end

%% IFFT
fprintf('Computing time-domain Green''s function...\n');

green_freq_full = [green_freq, conj(green_freq(:, end - 1:-1:2))];
green_time = real(ifft(green_freq_full, [], 2));

%% Plot
fprintf('Plotting results...\n');
RBD_plot_results(green_time, num_elements, fs, segment_num_samples, ...
    theta_vec, beam_power, theta_best, use_plane_wave, array_depths_m, ...
    green_freq, freq_hz, theta_plot_arrivals);
