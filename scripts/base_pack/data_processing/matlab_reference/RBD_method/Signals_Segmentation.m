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

scripts_dir = fileparts(script_dir);
project_dir = fileparts(scripts_dir);

cd(scripts_dir);
addpath(script_dir);
addpath(genpath(fullfile(scripts_dir, 'function')));
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
split_strategy = "periodic";    % "periodic" or "Range_nearby"
manual_dataset_variant_tag = "";    % optional extra suffix appended after auto tag

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
        split_options.gap_s = 15;
        % Use "before" or "after" the minimum range point for training.
        split_options.train_side = "after";
    otherwise
        error('Unsupported split_strategy: %s.', split_strategy);
end

% RBD feature extraction
theta_vec = linspace(-90, 90, 181) * pi / 180;   % [rad]
use_plane_wave = false;
normalize_spectrum = false;
rbd_beam_selection = "best";     % "best" or "multipath"
rbd_multipath_options = struct();
rbd_multipath_options.peak_threshold_db = -6;
rbd_multipath_options.min_separation_deg = 2;
rbd_multipath_options.max_num_peaks = Inf;
rbd_multipath_options.sidelobe_reject_db = 3;

% RBD frequency estimation and neural-network output selection.
% "full" saves all one-sided FFT bins.
% "selected" runs and saves only rbd_selected_frequency_modes.
rbd_frequency_estimation = "selected";   % "full" or "selected"
rbd_selected_frequency_modes = ["deep", "shallow"];  % "mel", "deep", "shallow", "adapt"

% Input labels
event_name = 'S5';
origindata_dir = DS_get_origindata_dir(project_dir);
range_file = fullfile(origindata_dir, 'events', 'range', ...
    'RangeEventS5', 'SproulToVLA.S5.txt');

%% Load common dataset inputs
[range_time_s, range_km_raw] = DS_load_range_labels(range_file);
[signal_time_full, array_depths_m, ~] = ...
    DS_load_vla_signals(project_dir, event_name, fs);
num_elements = numel(array_depths_m);

%% Environment for RBD
event_dir = fullfile(origindata_dir, 'events', event_name);
ctd_data = load(fullfile(event_dir, "CTD_i9605.mat"));
sound_speed_depth_m = ctd_data.T.depth_m;
sound_speed_ms = ctd_data.T.sound_speed_ms;
[sound_speed_depth_m, sound_speed_ms] = RBD_extend_sound_speed_profile( ...
    sound_speed_depth_m, sound_speed_ms, max(array_depths_m));

clear ctd_data event_dir origindata_dir;

%% Segment signals
segments = DS_build_segments(signal_time_full, fs, segment_duration_s, ...
    segment_step_s, segment_start_s, segment_end_s, range_time_s, range_km_raw);
segment_start_s = segments.segment_start_s;
segment_end_s = segments.segment_end_s;
segment_num_samples = segments.segment_num_samples;
segment_start_time_s = segments.segment_start_time_s;
segment_center_time_s = segments.segment_center_time_s;
segment_stop_time_s = segments.segment_stop_time_s;
segment_range_km = segments.segment_range_km;
valid_sample = segments.valid_sample;
segment_sample_start_idx = segments.segment_sample_start_idx;
segment_sample_stop_idx = segments.segment_sample_stop_idx;
num_segments = segments.num_segments;

%% RBD decomposition and neural-network HDF5 output
fprintf('Preparing steering delays...\n');

tau_matrix = RBD_compute_tau(theta_vec, array_depths_m, ...
    sound_speed_ms, sound_speed_depth_m, use_plane_wave);

rbd_options = {'NormalizeSpectrum', normalize_spectrum};
rbd_config = struct();
rbd_config.normalize_spectrum = logical(normalize_spectrum);
rbd_config.use_plane_wave = logical(use_plane_wave);
rbd_beam_selection = lower(strtrim(convertCharsToStrings(rbd_beam_selection)));
switch rbd_beam_selection
    case "best"
        multipath_beam = false;
    case "multipath"
        multipath_beam = true;
    otherwise
        error('Unsupported rbd_beam_selection: %s.', rbd_beam_selection);
end
rbd_options = [rbd_options, {'multipath_beam', multipath_beam}];
rbd_config.rbd_beam_selection = rbd_beam_selection;
rbd_config.multipath_beam = logical(multipath_beam);
rbd_frequency_estimation = lower(strtrim(convertCharsToStrings( ...
    rbd_frequency_estimation)));
switch rbd_frequency_estimation
    case "full"
        frequency_selection_modes = "full";
    case "selected"
        frequency_selection_modes = convertCharsToStrings( ...
            rbd_selected_frequency_modes(:).');
        if isempty(frequency_selection_modes)
            error('rbd_selected_frequency_modes must not be empty.');
        end
        if any(lower(strtrim(frequency_selection_modes)) == "full")
            error(['rbd_selected_frequency_modes must not contain "full"; ', ...
                'set rbd_frequency_estimation to "full" instead.']);
        end
    otherwise
        error('Unsupported rbd_frequency_estimation: %s.', ...
            rbd_frequency_estimation);
end
rbd_config.rbd_frequency_estimation = rbd_frequency_estimation;
rbd_config.rbd_selected_frequency_modes = frequency_selection_modes;
if rbd_config.multipath_beam
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
    rbd_config.rbd_multipath_options = rbd_multipath_options;
end

full_freq_hz = (0:floor(segment_num_samples / 2)) * fs / segment_num_samples;

% Leave empty for built-in defaults. Add fields here only to override defaults.
frequency_selection_config = struct();

dataset_variant_tag = RBD_make_dataset_variant_tag( ...
    frequency_selection_modes, frequency_selection_config, ...
    segment_duration_s, segment_step_s, normalize_spectrum, ...
    use_plane_wave, rbd_beam_selection, rbd_frequency_estimation);
manual_dataset_variant_tag = DS_sanitize_dataset_variant_tag(manual_dataset_variant_tag);
dataset_variant_tag = DS_append_dataset_variant_suffix( ...
    dataset_variant_tag, manual_dataset_variant_tag);

if any(lower(string(frequency_selection_modes)) == "adapt")
    frequency_selection_config.signal_time_full = signal_time_full;
    frequency_selection_config.segment_sample_start_idx = segment_sample_start_idx;
    frequency_selection_config.segment_sample_stop_idx = segment_sample_stop_idx;
    frequency_selection_config.adapt_candidate_segment_idx = find(valid_sample);
    frequency_selection_config.fs = fs;
    frequency_selection_config.snapshot_num_samples = segment_num_samples;
    frequency_selection_config.snapshot_step_samples = segment_num_samples;
end

[freq_bin_idx, freq_hz, frequency_selection_info] = DS_select_frequency_bins( ...
    full_freq_hz, frequency_selection_modes, frequency_selection_config);
switch rbd_frequency_estimation
    case "full"
        rbd_decompose_freq_bin_idx = [];
    case "selected"
        rbd_decompose_freq_bin_idx = freq_bin_idx;
end
rbd_options = [rbd_options, {'FrequencyBinIdx', rbd_decompose_freq_bin_idx}];
num_freq_bins = numel(freq_hz);
num_arrival_slots = numel(theta_vec);
rbd_config.num_arrival_slots = num_arrival_slots;
rbd_config.frequency_selection = 'combined_frequency_selection_modes';
rbd_config.frequency_selection_modes = frequency_selection_modes;
rbd_config.frequency_selection_info = frequency_selection_info;
rbd_config.selected_fft_bin_idx = uint32(freq_bin_idx);
rbd_config.selected_freq_hz = freq_hz;
rbd_config.full_num_freq_bins = numel(full_freq_hz);
rbd_config.manual_dataset_variant_tag = manual_dataset_variant_tag;

if save_results
    [split_indices, split_names, segment_split_idx, split_metadata] = ...
        SPL_build_split_indices(split_strategy, num_segments, valid_sample, ...
        segment_range_km, segment_center_time_s, segment_step_s, ...
        split_options);

    dataset_variant_tag = DS_sanitize_dataset_variant_tag(dataset_variant_tag);
    split_strategy_dir_name = SPL_make_split_strategy_dir_name( ...
        split_strategy, split_metadata);
    if strlength(dataset_variant_tag) > 0
        split_strategy_dir_name = sprintf('%s_%s', ...
            split_strategy_dir_name, dataset_variant_tag);
    end
    dataset_name = split_strategy_dir_name;
    recommended_model_name = "complex_cnn_range";
    train_command = sprintf(['python3 scripts_py/RBD_method/Network_main.py train ', ...
        '--model %s --data %s'], recommended_model_name, dataset_name);
    predict_command = sprintf(['python3 scripts_py/RBD_method/Network_main.py predict ', ...
        '--model %s --data %s'], recommended_model_name, dataset_name);

    results_dir = fullfile(project_dir, 'outputs', 'Datasets', ...
        split_strategy_dir_name);
    if ~isfolder(results_dir)
        mkdir(results_dir);
    end

    dataset_tag = SPL_make_dataset_tag(split_strategy, split_metadata, ...
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

        RBD_create_nn_h5(split_files{split_idx}, split_indices{split_idx}, ...
            num_elements, num_freq_bins, theta_vec, freq_hz, ...
            array_depths_m, range_time_s, range_km_raw, segment_range_km, ...
            valid_sample, segment_start_time_s, segment_center_time_s, ...
            segment_stop_time_s, segment_sample_start_idx, ...
            segment_sample_stop_idx, num_arrival_slots);

        fprintf('Writing %s neural-network HDF5 to %s (%d samples)\n', ...
            split_names{split_idx}, split_files{split_idx}, ...
            numel(split_indices{split_idx}));
    end

    RBD_write_dataset_metadata_json(metadata_file, split_strategy_dir_name, ...
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

        [green_freq, ~, rbd_result] = RBD_decompose( ...
            signal_time_seg, fs, theta_vec, tau_matrix, rbd_options{:});
        feature_freq_hz = rbd_result.freq_hz;
        if numel(feature_freq_hz) ~= num_freq_bins || ...
                any(abs(feature_freq_hz - freq_hz) > 10 * eps(max(freq_hz)))
            error('Feature frequency axis does not match expected frequency axis.');
        end

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

    fprintf('\nDataset name:\n  %s\n', dataset_name);
    fprintf('Recommended training command:\n  %s\n', train_command);
    fprintf('Recommended prediction command:\n  %s\n\n', predict_command);
end

clear signal_time_full signal_time_seg green_freq rbd_result green_feature segments;

fprintf('Signal segmentation complete.\n');

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
