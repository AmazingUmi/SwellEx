%% VLA element-ratio neural-network dataset generation
% Load VLA element time series, extract consecutive time segments, build
% pairwise element frequency-ratio features, and save them as NN inputs.
%
% Main dataset output:
%   - outputs/Datasets/<split_strategy>/*_train.h5 and *_test.h5
%   - outputs/Datasets/<split_strategy>/*_metadata.json
%   - /X: num_samples x num_elements x num_elements x num_freq_bins x 2
%         X(:,:,:,:,1) is real(ratio_freq)
%         X(:,:,:,:,2) is imag(ratio_freq)
%         ratio_freq(i,j,f)=FFT(element_i,f)/FFT(element_j,f)
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
segment_start_s = [];           % first segment start time [s]
% set [] for record start from the start of sig

segment_end_s   = [];           % last segment start time [s]
% set [] for full record to the end of sig

% Dataset output and split strategy
save_results = true;
split_strategy = "Range_nearby";    % "periodic" or "Range_nearby"
dataset_variant_tag = "elm_pairwise_ratio";   % user-controlled suffix

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

% Element-ratio feature extraction
denom_floor_relative = 1e-6;
feature_config = struct();
feature_config.mode = 'element_pairwise_frequency_ratio';
feature_config.definition = ...
    'ratio_freq(i,j,f)=FFT(element_i,f)/FFT(element_j,f)';
feature_config.denom_floor_relative = denom_floor_relative;
feature_config.h5_x_shape = ...
    'sample x numerator_element x denominator_element x frequency x real_imag';
feature_config.real_imag_index = ...
    'X(:,:,:,:,1)=real(ratio_freq), X(:,:,:,:,2)=imag(ratio_freq).';

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

%% Element-ratio feature and neural-network HDF5 output
freq_hz = (0:floor(segment_num_samples / 2)) * fs / segment_num_samples;
num_freq_bins = numel(freq_hz);

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
    file_stem = sprintf('ELM_ratio_freq_nn_S5_%s', dataset_tag);
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

        ELM_create_nn_h5(split_files{split_idx}, split_indices{split_idx}, ...
            num_elements, num_freq_bins, freq_hz, array_depths_m, ...
            range_time_s, range_km_raw, segment_range_km, valid_sample, ...
            segment_start_time_s, segment_center_time_s, ...
            segment_stop_time_s, segment_sample_start_idx, ...
            segment_sample_stop_idx);

        fprintf('Writing %s ELM HDF5 to %s (%d samples)\n', ...
            split_names{split_idx}, split_files{split_idx}, ...
            numel(split_indices{split_idx}));
    end

    ELM_write_dataset_metadata_json(metadata_file, split_strategy_dir_name, ...
        split_metadata, split_files, split_names, num_segments, fs, ...
        segment_duration_s, segment_step_s, segment_start_s, segment_end_s, ...
        range_file, freq_hz, array_depths_m, feature_config, ...
        dataset_variant_tag);

    fprintf('Building pairwise element-ratio features and streaming to HDF5...\n');

    for segment_idx = 1:num_segments
        split_idx = double(segment_split_idx(segment_idx));
        if split_idx == 0
            continue;
        end

        signal_time_seg = signal_time_full(:, ...
            segment_sample_start_idx(segment_idx):segment_sample_stop_idx(segment_idx));

        [ratio_freq, feature_freq_hz, feature_info] = ...
            ELM_extract_ratio_feature(signal_time_seg, fs, denom_floor_relative);
        if numel(feature_freq_hz) ~= num_freq_bins || ...
                any(abs(feature_freq_hz - freq_hz) > 10 * eps(max(freq_hz)))
            error('Feature frequency axis does not match expected frequency axis.');
        end

        ratio_feature = zeros(1, num_elements, num_elements, ...
            num_freq_bins, 2, 'single');
        ratio_feature(1, :, :, :, 1) = single(real(ratio_freq));
        ratio_feature(1, :, :, :, 2) = single(imag(ratio_freq));

        split_write_counts(split_idx) = split_write_counts(split_idx) + 1;
        local_idx = split_write_counts(split_idx);
        total_written_samples = total_written_samples + 1;
        h5_file = split_files{split_idx};

        h5write(h5_file, '/X', ratio_feature, ...
            [local_idx, 1, 1, 1, 1], ...
            [1, num_elements, num_elements, num_freq_bins, 2]);

        if mod(total_written_samples, max(1, floor(total_split_samples / 10))) == 0 || ...
                total_written_samples == total_split_samples
            fprintf(['  %d/%d selected windows complete. Range = %.3f km. ', ...
                'Denom floor = %.3e\n'], ...
                total_written_samples, total_split_samples, ...
                segment_range_km(segment_idx), feature_info.denom_floor);
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

clear signal_time_full signal_time_seg ratio_freq ratio_feature segments;

fprintf('ELM signal segmentation complete.\n');
