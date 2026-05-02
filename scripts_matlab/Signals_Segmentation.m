%% VLA RBD neural-network dataset generation
% Load VLA element time series, extract consecutive time segments, run RBD,
% and save frequency-domain Green's functions as neural-network inputs.
%
% Main dataset output:
%   - outputs/Datasets/<split_strategy>/*_train.h5 and *_test.h5
%   - outputs/Datasets/<split_strategy>/*_metadata.json
%   - /X: num_samples x num_elements x num_freq_bins x 2
%         X(:,:,:,1) is real(green_freq), X(:,:,:,2) is imag(green_freq)
%   - /y_range_km: range label for each segment
%   - /valid_sample: 1 if y_range_km is finite, otherwise 0

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
% Candidate signal segmentation range used before applying split_strategy.
fs = 1500;                      % [Hz]
segment_duration_s = 1.0;       % segment duration [s]
segment_step_s  = 1.0;          % time step between adjacent segments [s]
segment_start_s = [];           % first segment start time [s]; 
% set [] for record start from the start of sig

segment_end_s   = [];           % last segment start time [s];  
% set [] for full record to the end of sig

% Dataset output and split strategy
save_results = true;
split_strategy = "Range_nearby";    % "periodic" or "Range_nearby"
dataset_variant_tag = "before_singlepath_nonormal";           % user-controlled suffix for output files

switch split_strategy
    case "periodic"
        split_options = struct();
        % Deterministic split: 4 train windows, then 1 test window.
        split_options.train_test_ratio = [4 1];
    case "Range_nearby"
        split_options = struct();
        % Take symmetric windows around the minimum range point.
        split_options.half_duration_s = 800;
        % Skip windows nearest to the minimum range point.
        split_options.gap_s = 30;
        % Use "before" or "after" the minimum range point for training.
        split_options.train_side = "before";
    otherwise
        error('Unsupported split_strategy: %s.', split_strategy);
end

% RBD feature extraction
theta_vec = linspace(-90, 90, 181) * pi / 180;   % [rad]
use_plane_wave = false;
normalize_spectrum = false;
multipath_beam = false;
if multipath_beam
    multipath_peak_threshold_db = -6;
    multipath_min_separation_deg = 2;
    multipath_max_num_peaks = Inf;
    multipath_sidelobe_reject_db = 3;
end

% Input labels
range_file = fullfile(project_dir, 'events', 'range', ...
    'RangeEventS5', 'SproulToVLA.S5.txt');

%% Array geometry
position_file = fullfile(project_dir, 'positions', 'positions_vla.txt');
position_table = readmatrix(position_file);   % [channel_index, depth_m]
array_depths_m = flip(position_table(:, 2).');
num_elements = numel(array_depths_m);

clear position_file position_table;

%% Environment for RBD
event_dir = fullfile(project_dir, 'events', 'S5');
ctd_data = load(fullfile(event_dir, "CTD_i9605.mat"));
sound_speed_depth_m = ctd_data.T.depth_m;
sound_speed_ms = ctd_data.T.sound_speed_ms;
[sound_speed_depth_m, sound_speed_ms] = extend_sound_speed_profile( ...
    sound_speed_depth_m, sound_speed_ms, max(array_depths_m));

clear ctd_data;

%% Load range labels
range_data = readmatrix(range_file, 'FileType', 'text', ...
    'NumHeaderLines', 1);
if size(range_data, 2) < 4
    error('Range file must contain at least 4 columns: Jday Time Duration Range(km).');
end

range_time_s = range_data(:, 3).' * 60;
range_km_raw = range_data(:, 4).';
range_valid = isfinite(range_time_s) & isfinite(range_km_raw);
range_time_s = range_time_s(range_valid);
range_km_raw = range_km_raw(range_valid);

if numel(range_time_s) < 2
    error('Range file must contain at least two finite range samples.');
end

[range_time_s, range_sort_idx] = sort(range_time_s);
range_km_raw = range_km_raw(range_sort_idx);

clear range_data range_valid range_sort_idx;

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

clear first_channel_data first_channel_signal first_channel_file;
clear channel_data channel_signal channel_idx channel_file channel_data_dir event_dir;

%% Segment signals
segment_num_samples = round(fs * segment_duration_s);

if isempty(segment_start_s)
    segment_start_s = 0;
end

if isempty(segment_end_s)
    segment_end_s = record_duration_s - segment_duration_s;
end

segment_start_s = max(segment_start_s, 0);
segment_end_s = min(segment_end_s, record_duration_s - segment_duration_s);
if segment_end_s < segment_start_s
    error(['Invalid segmentation range: segment_start_s=%.3f s, ', ...
        'segment_end_s=%.3f s, record_duration_s=%.3f s.'], ...
        segment_start_s, segment_end_s, record_duration_s);
end

segment_start_time_s = segment_start_s:segment_step_s:segment_end_s;
segment_center_time_s = segment_start_time_s + segment_duration_s / 2;
num_segments = numel(segment_start_time_s);
segment_time_rel_s = (0:segment_num_samples - 1) / fs;
segment_stop_time_s = segment_start_time_s + segment_duration_s;
segment_range_km = interp1(range_time_s, range_km_raw, ...
    segment_center_time_s, 'linear', NaN);
valid_sample = isfinite(segment_range_km);

segment_sample_start_idx = zeros(1, num_segments);
segment_sample_stop_idx = zeros(1, num_segments);

fprintf(['Loaded %d elements, %.2f s record. Extracting %d segments from ', ...
    '%.2f s to %.2f s.\n'], ...
    num_elements, record_duration_s, num_segments, ...
    segment_start_time_s(1), segment_start_time_s(end));

for segment_idx = 1:num_segments
    sample_start_idx = round(segment_start_time_s(segment_idx) * fs) + 1;
    sample_stop_idx = sample_start_idx + segment_num_samples - 1;

    segment_sample_start_idx(segment_idx) = sample_start_idx;
    segment_sample_stop_idx(segment_idx) = sample_stop_idx;
end

clear sample_start_idx sample_stop_idx segment_idx;

%% RBD decomposition and neural-network HDF5 output
fprintf('Preparing steering delays...\n');

tau_matrix = compute_tau(theta_vec, array_depths_m, ...
    sound_speed_ms, sound_speed_depth_m, use_plane_wave);

rbd_options = {'NormalizeSpectrum', normalize_spectrum, ...
    'multipath_beam', multipath_beam};
rbd_config = struct();
rbd_config.normalize_spectrum = logical(normalize_spectrum);
rbd_config.use_plane_wave = logical(use_plane_wave);
rbd_config.multipath_beam = logical(multipath_beam);
if multipath_beam
    rbd_options = [rbd_options, { ...
        'MultipathPeakThresholdDb', multipath_peak_threshold_db, ...
        'MultipathMinSeparationDeg', multipath_min_separation_deg, ...
        'MultipathMaxNumPeaks', multipath_max_num_peaks, ...
        'MultipathSidelobeRejectDb', multipath_sidelobe_reject_db}];
    rbd_config.multipath_peak_threshold_db = multipath_peak_threshold_db;
    rbd_config.multipath_min_separation_deg = multipath_min_separation_deg;
    rbd_config.multipath_max_num_peaks = multipath_max_num_peaks;
    rbd_config.multipath_sidelobe_reject_db = multipath_sidelobe_reject_db;
end

freq_hz = (0:floor(segment_num_samples / 2)) * fs / segment_num_samples;
num_freq_bins = numel(freq_hz);
num_arrival_slots = numel(theta_vec);
rbd_config.num_arrival_slots = num_arrival_slots;

if save_results
    [split_indices, split_names, segment_split_idx, split_metadata] = ...
        SS_build_split_indices(split_strategy, num_segments, valid_sample, ...
        segment_range_km, segment_center_time_s, segment_step_s, ...
        split_options);

    dataset_variant_tag = sanitize_dataset_variant_tag(dataset_variant_tag);
    split_strategy_dir_name = SS_make_split_strategy_dir_name( ...
        split_strategy, split_metadata);
    if strlength(dataset_variant_tag) > 0
        split_strategy_dir_name = sprintf('%s_%s', ...
            split_strategy_dir_name, dataset_variant_tag);
    end
    results_dir = fullfile(project_dir, 'outputs', 'Datasets', ...
        split_strategy_dir_name);
    if ~isfolder(results_dir)
        mkdir(results_dir);
    end

    dataset_tag = SS_make_dataset_tag(split_strategy, split_metadata, ...
        segment_start_s, segment_end_s, segment_step_s);
    if strlength(dataset_variant_tag) > 0
        dataset_tag = sprintf('%s_%s', dataset_tag, dataset_variant_tag);
    end
    file_stem = sprintf('RBD_green_freq_nn_S5_%s', dataset_tag);
    metadata_file = fullfile(results_dir, sprintf('%s_metadata.json', file_stem));
    split_files = cell(1, numel(split_names));
    for split_idx = 1:numel(split_names)
        split_files{split_idx} = fullfile(results_dir, sprintf( ...
            '%s_%s.h5', file_stem, split_names{split_idx}));
    end
    split_write_counts = zeros(1, numel(split_files));
    total_split_samples = sum(cellfun(@numel, split_indices));
    total_written_samples = 0;

    for split_idx = 1:numel(split_files)
        if isempty(split_indices{split_idx})
            continue;
        end

        SS_create_nn_h5(split_files{split_idx}, split_indices{split_idx}, ...
            num_elements, num_freq_bins, theta_vec, freq_hz, ...
            array_depths_m, range_time_s, range_km_raw, segment_range_km, ...
            valid_sample, segment_start_time_s, segment_center_time_s, ...
            segment_stop_time_s, segment_sample_start_idx, ...
            segment_sample_stop_idx, num_arrival_slots);

        fprintf('Writing %s neural-network HDF5 to %s (%d samples)\n', ...
            split_names{split_idx}, split_files{split_idx}, ...
            numel(split_indices{split_idx}));
    end

    SS_write_dataset_metadata_json(metadata_file, split_strategy_dir_name, ...
        split_metadata, split_files, split_names, num_segments, fs, ...
        segment_duration_s, segment_step_s, ...
        segment_start_s, segment_end_s, normalize_spectrum, use_plane_wave, ...
        range_file, theta_vec, freq_hz, array_depths_m, rbd_config, ...
        dataset_variant_tag);

    fprintf('Running RBD and streaming train/test X/y to HDF5...\n');

    for segment_idx = 1:num_segments
        split_idx = double(segment_split_idx(segment_idx));
        if split_idx == 0
            continue;
        end

        signal_time_seg = signal_time_full(:, ...
            segment_sample_start_idx(segment_idx):segment_sample_stop_idx(segment_idx));

        [green_freq, ~, rbd_result] = rbd_decompose( ...
            signal_time_seg, fs, theta_vec, tau_matrix, rbd_options{:});

        green_feature = zeros(1, num_elements, num_freq_bins, 2, 'single');
        green_feature(1, :, :, 1) = single(real(green_freq));
        green_feature(1, :, :, 2) = single(imag(green_freq));

        split_write_counts(split_idx) = split_write_counts(split_idx) + 1;
        local_idx = split_write_counts(split_idx);
        total_written_samples = total_written_samples + 1;
        h5_file = split_files{split_idx};

        h5write(h5_file, '/X', green_feature, ...
            [local_idx, 1, 1, 1], [1, num_elements, num_freq_bins, 2]);
        h5write(h5_file, '/rbd/theta_best_rad', rbd_result.theta_best, ...
            [local_idx, 1], [1, 1]);
        [theta_selected_row, selected_beam_power_row, num_selected_angles] = ...
            make_arrival_rows(rbd_result, num_arrival_slots);
        h5write(h5_file, '/rbd/num_selected_angles', ...
            uint16(num_selected_angles), [local_idx, 1], [1, 1]);
        h5write(h5_file, '/rbd/theta_selected_rad', theta_selected_row, ...
            [local_idx, 1], [1, num_arrival_slots]);
        h5write(h5_file, '/rbd/selected_beam_power', ...
            single(selected_beam_power_row), ...
            [local_idx, 1], [1, num_arrival_slots]);
        h5write(h5_file, '/rbd/beam_power', single(rbd_result.beam_power(:).'), ...
            [local_idx, 1], [1, numel(theta_vec)]);
        h5write(h5_file, '/rbd/signal_freq_scale', ...
            rbd_result.signal_freq_scale, [local_idx, 1], [1, 1]);

        if mod(total_written_samples, max(1, floor(total_split_samples / 10))) == 0 || ...
                total_written_samples == total_split_samples
            fprintf('  %d/%d selected windows complete. Range = %.3f km\n', ...
                total_written_samples, total_split_samples, ...
                segment_range_km(segment_idx));
        end
    end

    expected_write_counts = cellfun(@numel, split_indices);
    if any(split_write_counts ~= expected_write_counts)
        error('Split write counts do not match expected split sample counts.');
    end

    for split_idx = 1:numel(split_files)
        fprintf('Saved %s dataset to %s\n', ...
            split_names{split_idx}, split_files{split_idx});
    end
end

clear signal_time_full signal_time_seg green_freq rbd_result green_feature;

fprintf('Signal segmentation complete.\n');

function dataset_variant_tag = sanitize_dataset_variant_tag(dataset_variant_tag)
dataset_variant_tag = string(dataset_variant_tag);
dataset_variant_tag = strtrim(dataset_variant_tag);

if strlength(dataset_variant_tag) == 0
    return;
end

dataset_variant_tag = regexprep(dataset_variant_tag, '[^A-Za-z0-9_-]+', '_');
dataset_variant_tag = regexprep(dataset_variant_tag, '_+', '_');
dataset_variant_tag = regexprep(dataset_variant_tag, '^_|_$', '');
end

function [theta_selected_row, selected_beam_power_row, num_selected_angles] = ...
    make_arrival_rows(rbd_result, num_arrival_slots)

theta_selected_row = NaN(1, num_arrival_slots);
selected_beam_power_row = NaN(1, num_arrival_slots);

[~, arrival_sort_idx] = sort(rbd_result.selected_beam_power, 'descend');
num_selected_angles = min(numel(arrival_sort_idx), num_arrival_slots);

if num_selected_angles == 0
    return;
end

arrival_sort_idx = arrival_sort_idx(1:num_selected_angles);
theta_selected_row(1:num_selected_angles) = ...
    rbd_result.theta_selected(arrival_sort_idx);
selected_beam_power_row(1:num_selected_angles) = ...
    rbd_result.selected_beam_power(arrival_sort_idx);
end
