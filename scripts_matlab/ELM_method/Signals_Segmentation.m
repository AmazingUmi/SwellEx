%% VLA element-ratio neural-network dataset generation
% Load VLA element time series, extract consecutive time segments, build
% pairwise element frequency-ratio features, and save them as NN inputs.
%
% Main dataset output:
%   - outputs/Datasets/<split_strategy>/*_train.h5 and *_test.h5
%   - outputs/Datasets/<split_strategy>/*_metadata.json
%   - /X: num_samples x num_pairs x num_freq_bins x 2
%         X(:,:,:,1) is real(ratio_freq)
%         X(:,:,:,2) is imag(ratio_freq)
%         ratio_freq(i,j,f)=FFT(element_i,f)/FFT(element_j,f)
%         pairs are strict upper-triangle element pairs with i < j
%         frequency can be the full one-sided FFT axis or Mel-spaced bins
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
split_strategy = "periodic";    % "periodic" or "Range_nearby"
dataset_variant_tag = "elm_pairwise_ratio_upper_mel64";   % user-controlled suffix

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
        split_options.train_side = "before";
    otherwise
        error('Unsupported split_strategy: %s.', split_strategy);
end

% Element-ratio feature extraction
denom_floor_relative = 1e-6;
use_mel_frequency_selection = true;
mel_num_bins = 64;
mel_min_freq_hz = 1;             % avoid the DC bin by default
mel_max_freq_hz = fs / 2;

feature_config = struct();
feature_config.mode = 'element_pairwise_frequency_ratio';
feature_config.definition = ...
    'ratio_freq(i,j,f)=FFT(element_i,f)/FFT(element_j,f)';
feature_config.output_layout = 'strict_upper_pair_vector';
feature_config.denom_floor_relative = denom_floor_relative;
feature_config.frequency_selection = 'mel_nearest_fft_bin';
feature_config.use_mel_frequency_selection = use_mel_frequency_selection;
feature_config.mel_num_bins_requested = mel_num_bins;
feature_config.mel_min_freq_hz = mel_min_freq_hz;
feature_config.mel_max_freq_hz = mel_max_freq_hz;
feature_config.h5_x_shape = ...
    'sample x pair x frequency x real_imag';
feature_config.real_imag_index = ...
    'X(:,:,:,1)=real(ratio_freq), X(:,:,:,2)=imag(ratio_freq).';

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
[pair_i, pair_j] = ELM_make_upper_pair_indices(num_elements);
num_pairs = numel(pair_i);
feature_config.num_pairs = num_pairs;
feature_config.pair_selection = 'strict_upper_triangle_i_lt_j';
feature_config.pair_numerator_element_idx = pair_i;
feature_config.pair_denominator_element_idx = pair_j;

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
full_freq_hz = (0:floor(segment_num_samples / 2)) * fs / segment_num_samples;
if use_mel_frequency_selection
    [freq_bin_idx, freq_hz, mel_center_freq_hz] = ELM_make_mel_frequency_bins( ...
        full_freq_hz, mel_num_bins, mel_min_freq_hz, mel_max_freq_hz);
else
    freq_bin_idx = 1:numel(full_freq_hz);
    freq_hz = full_freq_hz;
    mel_center_freq_hz = [];
end
num_freq_bins = numel(freq_hz);
feature_config.num_freq_bins = num_freq_bins;
feature_config.selected_fft_bin_idx = uint32(freq_bin_idx);
feature_config.selected_freq_hz = freq_hz;
feature_config.mel_center_freq_hz = mel_center_freq_hz;

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
            num_pairs, num_freq_bins, freq_hz, array_depths_m, pair_i, pair_j, ...
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
            ELM_extract_ratio_feature(signal_time_seg, fs, ...
            denom_floor_relative, freq_bin_idx, pair_i, pair_j);
        if numel(feature_freq_hz) ~= num_freq_bins || ...
                any(abs(feature_freq_hz - freq_hz) > 10 * eps(max(freq_hz)))
            error('Feature frequency axis does not match expected frequency axis.');
        end

        ratio_feature = zeros(1, num_pairs, num_freq_bins, 2, 'single');
        ratio_feature(1, :, :, 1) = single(real(ratio_freq));
        ratio_feature(1, :, :, 2) = single(imag(ratio_freq));

        split_write_counts(split_idx) = split_write_counts(split_idx) + 1;
        local_idx = split_write_counts(split_idx);
        total_written_samples = total_written_samples + 1;
        h5_file = split_files{split_idx};

        h5write(h5_file, '/X', ratio_feature, ...
            [local_idx, 1, 1, 1], ...
            [1, num_pairs, num_freq_bins, 2]);

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
