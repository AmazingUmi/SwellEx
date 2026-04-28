%% RBD time-series analysis
% Sliding-window Ray-Based Deconvolution for continuous time analysis.
%
% This script analyzes consecutive VLA signal segments. Segment extraction
% only is handled by Signals_Segmentation.m.
%
% Main outputs:
%   - Best beam angle versus analysis time
%   - Time-domain Green's function versus analysis time for selected
%     receiver elements
%   - Element-wise Green's function peak amplitude and peak delay over time

%% Environment setup
clear; close all; clc;

try
    tmp = matlab.desktop.editor.getActive;
    script_dir = fileparts(tmp.Filename);
catch
    script_dir = fileparts(mfilename('fullpath'));
end

project_dir = fileparts(script_dir);

cd(script_dir);
addpath(script_dir);
addpath(fullfile(script_dir, 'RBD_function'));
clear tmp;

%% User parameters
fs = 1500;                         % [Hz]
segment_duration_s = 1.0;          % analysis-window duration [s]
segment_step_s = 1.0;              % time step between adjacent windows [s]
segment_start_s = 300.0;           % first window start time [s]
segment_end_s = 400.0;             % last window start time [s]; set [] for full record

theta_vec = linspace(-90, 90, 181) * pi / 180;   % [rad]
use_plane_wave = false;

green_plot_duration_s = 0.10;      % display first 0.10 s of g_e [s]
selected_element_idx = [1 6 11 16 21];
save_results = true;

%% Array and environment
position_file = fullfile(project_dir, 'positions', 'positions_vla.txt');
position_table = readmatrix(position_file);   % [channel_index, depth_m]
array_depths_m = flip(position_table(:, 2).');
num_elements = numel(array_depths_m);

selected_element_idx = selected_element_idx( ...
    selected_element_idx >= 1 & selected_element_idx <= num_elements);
if isempty(selected_element_idx)
    error(['selected_element_idx must contain at least one valid array ', ...
        'element index from 1 to %d.'], num_elements);
end

event_dir = fullfile(project_dir, 'events', 'S5');
ctd_data = load(fullfile(event_dir, "CTD_i9605.mat"));
sound_speed_depth_m = ctd_data.T.depth_m;
sound_speed_ms = ctd_data.T.sound_speed_ms;
[sound_speed_depth_m, sound_speed_ms] = extend_sound_speed_profile( ...
    sound_speed_depth_m, sound_speed_ms, max(array_depths_m));

clear position_file position_table ctd_data;

%% Load time-domain data
fprintf('Loading VLA time series...\n');

channel_data_dir = fullfile(event_dir, "vla_matfiles");

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

signal_time_full = signal_time_full - mean(signal_time_full, 2);

record_num_samples = size(signal_time_full, 2);
record_duration_s = record_num_samples / fs;

if isempty(segment_end_s)
    segment_end_s = record_duration_s - segment_duration_s;
end

segment_start_s = max(segment_start_s, 0);
segment_end_s = min(segment_end_s, record_duration_s - segment_duration_s);
if segment_end_s < segment_start_s
    error(['Invalid analysis range: segment_start_s=%.3f s, ', ...
        'segment_end_s=%.3f s, record_duration_s=%.3f s.'], ...
        segment_start_s, segment_end_s, record_duration_s);
end

window_start_time_s = segment_start_s:segment_step_s:segment_end_s;
window_center_time_s = window_start_time_s + segment_duration_s / 2;
num_windows = numel(window_start_time_s);
segment_num_samples = round(fs * segment_duration_s);

fprintf(['Loaded %d elements, %.2f s record. Running %d windows from ', ...
    '%.2f s to %.2f s.\n'], ...
    num_elements, record_duration_s, num_windows, ...
    window_start_time_s(1), window_start_time_s(end));

clear first_channel_data first_channel_signal first_channel_file;
clear channel_data channel_signal channel_idx channel_file channel_data_dir;

%% Steering delays
fprintf('Preparing steering delays...\n');

tau_matrix = compute_tau(theta_vec, array_depths_m, ...
    sound_speed_ms, sound_speed_depth_m, use_plane_wave);

green_num_samples = min(segment_num_samples, round(fs * green_plot_duration_s));
green_delay_s = (0:green_num_samples - 1) / fs;

theta_best_history = zeros(1, num_windows);
beam_power_history = zeros(numel(theta_vec), num_windows);
green_selected = zeros(numel(selected_element_idx), green_num_samples, num_windows);
green_peak_amp = zeros(num_elements, num_windows);
green_peak_delay_s = zeros(num_elements, num_windows);

%% Sliding-window RBD
fprintf('Running sliding-window RBD...\n');

for window_idx = 1:num_windows
    sample_start_idx = round(window_start_time_s(window_idx) * fs) + 1;
    sample_stop_idx = sample_start_idx + segment_num_samples - 1;

    signal_time_seg = signal_time_full(:, sample_start_idx:sample_stop_idx);

    [green_freq, ~, rbd_result] = rbd_decompose( ...
        signal_time_seg, fs, theta_vec, tau_matrix);

    green_freq_full = [green_freq, conj(green_freq(:, end - 1:-1:2))];
    green_time = real(ifft(green_freq_full, [], 2));

    theta_best_history(window_idx) = rbd_result.theta_best;
    beam_power_history(:, window_idx) = rbd_result.beam_power(:);
    green_selected(:, :, window_idx) = green_time(selected_element_idx, 1:green_num_samples);

    green_abs = abs(green_time(:, 1:green_num_samples));
    [green_peak_amp(:, window_idx), peak_idx] = max(green_abs, [], 2);
    green_peak_delay_s(:, window_idx) = green_delay_s(peak_idx);

    if mod(window_idx, max(1, floor(num_windows / 10))) == 0 || window_idx == num_windows
        fprintf('  %d/%d windows complete. Current best angle = %.2f deg\n', ...
            window_idx, num_windows, rbd_result.theta_best * 180 / pi);
    end
end

clear signal_time_seg rbd_result green_freq green_freq_full;
clear green_time green_abs peak_idx window_idx;
clear sample_start_idx sample_stop_idx;

%% Plot results
fprintf('Plotting time-series results...\n');

result = struct();
result.window_center_time_s = window_center_time_s;
result.t_window_center = window_center_time_s;
result.theta_vec = theta_vec;
result.theta_best_history = theta_best_history;
result.beam_power_history = beam_power_history;
result.B_power_history = beam_power_history;
result.green_delay_s = green_delay_s;
result.green_delay = green_delay_s;
result.green_selected = green_selected;
result.selected_element_idx = selected_element_idx;
result.selected_elements = selected_element_idx;
result.green_peak_amp = green_peak_amp;
result.green_peak_delay_s = green_peak_delay_s;
result.green_peak_delay = green_peak_delay_s;
result.num_elements = num_elements;
result.N = num_elements;
result.use_plane_wave = use_plane_wave;

plot_results_series(result);

%% Save results
if save_results
    results_dir = fullfile(script_dir, 'RBD_results');
    if ~isfolder(results_dir)
        mkdir(results_dir);
    end

    result_file = fullfile(results_dir, sprintf( ...
        'RBD_time_series_%06.1f_%06.1f_step_%04.1f.mat', ...
        segment_start_s, segment_end_s, segment_step_s));
    save(result_file, ...
        'fs', 'segment_duration_s', 'segment_step_s', ...
        'segment_start_s', 'segment_end_s', ...
        'window_start_time_s', 'window_center_time_s', ...
        'theta_vec', 'theta_best_history', 'beam_power_history', ...
        'selected_element_idx', 'green_delay_s', 'green_selected', ...
        'green_peak_amp', 'green_peak_delay_s', ...
        'use_plane_wave', '-v7.3');

    fprintf('Saved results to %s\n', result_file);
end

fprintf('RBD time-series analysis complete.\n');
