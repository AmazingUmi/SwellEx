%% VLA SCM neural-network dataset generation
% Load VLA element time series, extract consecutive time segments, build
% normalized spatial covariance matrix features, and save them as NN inputs.
%
% Main dataset output:
%   - outputs/Datasets/<split_strategy>/*_train.h5 and *_test.h5
%   - outputs/Datasets/<split_strategy>/*_metadata.json
%   - /X: num_samples x num_pairs x num_freq_bins x 2
%         X(:,:,:,1) is real(SCM)
%         X(:,:,:,2) is imag(SCM)
%         pairs are upper-triangle element pairs with i <= j
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
fs = 1500;                      % [Hz]

% Base 1-second snapshots. SCM samples are built by averaging Ns consecutive
% snapshots; adjacent SCM samples share snapshot_overlap_count snapshots.
segment_duration_s = 1.0;       % one SCM snapshot length [s]
segment_step_s  = 1.0;          % time step between adjacent snapshots [s]
num_snapshots_per_segment = 1;   % Ns
snapshot_overlap_count = 0;      % first SCM: 1-4 s, second SCM: 2-5 s
segment_start_s = [];           % first segment start time [s]
segment_end_s   = [];           % last segment start time [s]

% Dataset output and split strategy
save_results = true;
split_strategy = "Range_nearby";    % "periodic" or "Range_nearby"
dataset_variant_tag = "scm_upper_diag_mel64_snap1s_ns4_ov3";

switch split_strategy
    case "periodic"
        split_options = struct();
        split_options.train_test_ratio = [4 1];
    case "Range_nearby"
        split_options = struct();
        split_options.half_duration_s = 800;
        split_options.gap_s = 15;
        split_options.train_side = "before";
        % SCM samples overlap by 3 seconds; use non-overlapping test samples
        % for an effective 4:1 train/test density on opposite sides.
        split_options.test_step_s = 4;
    otherwise
        error('Unsupported split_strategy: %s.', split_strategy);
end

% SCM feature extraction
norm_floor = 1.0e-12;
use_mel_frequency_selection = true;
mel_num_bins = 64;
mel_min_freq_hz = 1;
mel_max_freq_hz = fs / 2;

feature_config = struct();
feature_config.mode = 'spatial_covariance_matrix';
feature_config.definition = ...
    'C_q=mean_s((x_s/||x_s||)(x_s/||x_s||)^H)';
feature_config.output_layout = 'upper_triangle_pair_vector_with_diagonal';
feature_config.snapshot_duration_s = segment_duration_s;
feature_config.snapshot_step_s = segment_step_s;
feature_config.num_snapshots_per_segment = num_snapshots_per_segment;
feature_config.snapshot_overlap_count = snapshot_overlap_count;
feature_config.norm_floor = norm_floor;
feature_config.frequency_selection = 'mel_nearest_fft_bin';
feature_config.use_mel_frequency_selection = use_mel_frequency_selection;
feature_config.mel_num_bins_requested = mel_num_bins;
feature_config.mel_min_freq_hz = mel_min_freq_hz;
feature_config.mel_max_freq_hz = mel_max_freq_hz;
feature_config.h5_x_shape = ...
    'sample x pair x frequency x real_imag';
feature_config.real_imag_index = ...
    'X(:,:,:,1)=real(scm_feature), X(:,:,:,2)=imag(scm_feature).';

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
[pair_i, pair_j] = SCM_make_upper_pair_indices(num_elements);
num_pairs = numel(pair_i);
feature_config.num_pairs = num_pairs;
feature_config.pair_selection = 'upper_triangle_i_le_j';
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

scm_step_segments = num_snapshots_per_segment - snapshot_overlap_count;
if scm_step_segments < 1
    error('snapshot_overlap_count must be less than num_snapshots_per_segment.');
end
if num_segments < num_snapshots_per_segment
    error('Not enough 1-second segments to build SCM samples.');
end
scm_start_segment_idx = 1:scm_step_segments:(num_segments - num_snapshots_per_segment + 1);
scm_stop_segment_idx = scm_start_segment_idx + num_snapshots_per_segment - 1;
num_scm_segments = numel(scm_start_segment_idx);
scm_segment_step_s = scm_step_segments * segment_step_s;
scm_segment_duration_s = segment_duration_s + ...
    (num_snapshots_per_segment - 1) * segment_step_s;

scm_segment_start_time_s = segment_start_time_s(scm_start_segment_idx);
scm_segment_stop_time_s = segment_stop_time_s(scm_stop_segment_idx);
scm_segment_center_time_s = ...
    (scm_segment_start_time_s + scm_segment_stop_time_s) / 2;
scm_segment_range_km = interp1(range_time_s, range_km_raw, ...
    scm_segment_center_time_s, 'linear', NaN);
scm_valid_sample = isfinite(scm_segment_range_km);
for scm_idx = 1:num_scm_segments
    base_idx = scm_start_segment_idx(scm_idx):scm_stop_segment_idx(scm_idx);
    scm_valid_sample(scm_idx) = scm_valid_sample(scm_idx) && all(valid_sample(base_idx));
end
scm_segment_sample_start_idx = segment_sample_start_idx(scm_start_segment_idx);
scm_segment_sample_stop_idx = segment_sample_stop_idx(scm_stop_segment_idx);

snapshot_num_samples = segment_num_samples;
snapshot_step_samples = round(segment_step_s * fs);
feature_config.scm_step_segments = scm_step_segments;
feature_config.scm_segment_duration_s = scm_segment_duration_s;
feature_config.snapshot_num_samples = snapshot_num_samples;
feature_config.snapshot_step_samples = snapshot_step_samples;
feature_config.actual_num_snapshots = num_snapshots_per_segment;

%% SCM feature and neural-network HDF5 output
full_freq_hz = (0:floor(snapshot_num_samples / 2)) * fs / snapshot_num_samples;
if use_mel_frequency_selection
    [freq_bin_idx, freq_hz, mel_center_freq_hz] = SCM_make_mel_frequency_bins( ...
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
        SPL_build_split_indices(split_strategy, num_scm_segments, scm_valid_sample, ...
        scm_segment_range_km, scm_segment_center_time_s, scm_segment_step_s, ...
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
        segment_start_s, segment_end_s, scm_segment_step_s);
    if strlength(dataset_variant_tag) > 0
        dataset_tag = sprintf('%s_%s', dataset_tag, dataset_variant_tag);
    end
    file_stem = sprintf('SCM_nn_S5_%s', dataset_tag);
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

        SCM_create_nn_h5(split_files{split_idx}, split_indices{split_idx}, ...
            num_pairs, num_freq_bins, freq_hz, array_depths_m, pair_i, pair_j, ...
            range_time_s, range_km_raw, scm_segment_range_km, scm_valid_sample, ...
            scm_segment_start_time_s, scm_segment_center_time_s, ...
            scm_segment_stop_time_s, scm_segment_sample_start_idx, ...
            scm_segment_sample_stop_idx);

        fprintf('Writing %s SCM HDF5 to %s (%d samples)\n', ...
            split_names{split_idx}, split_files{split_idx}, ...
            numel(split_indices{split_idx}));
    end

    SCM_write_dataset_metadata_json(metadata_file, split_strategy_dir_name, ...
        split_metadata, split_files, split_names, num_scm_segments, fs, ...
        scm_segment_duration_s, scm_segment_step_s, segment_start_s, segment_end_s, ...
        range_file, freq_hz, array_depths_m, feature_config, ...
        dataset_variant_tag);

    fprintf('Building SCM features and streaming to HDF5...\n');

    for scm_idx = 1:num_scm_segments
        split_idx = double(segment_split_idx(scm_idx));
        if split_idx == 0
            continue;
        end

        signal_time_seg = signal_time_full(:, ...
            scm_segment_sample_start_idx(scm_idx):scm_segment_sample_stop_idx(scm_idx));

        [scm_freq, feature_freq_hz, feature_info] = ...
            SCM_extract_feature(signal_time_seg, fs, snapshot_num_samples, ...
            snapshot_step_samples, freq_bin_idx, pair_i, pair_j, norm_floor);
        if numel(feature_freq_hz) ~= num_freq_bins || ...
                any(abs(feature_freq_hz - freq_hz) > 10 * eps(max(freq_hz)))
            error('Feature frequency axis does not match expected frequency axis.');
        end

        scm_feature = zeros(1, num_pairs, num_freq_bins, 2, 'single');
        scm_feature(1, :, :, 1) = single(real(scm_freq));
        scm_feature(1, :, :, 2) = single(imag(scm_freq));

        split_write_counts(split_idx) = split_write_counts(split_idx) + 1;
        local_idx = split_write_counts(split_idx);
        total_written_samples = total_written_samples + 1;
        h5_file = split_files{split_idx};

        h5write(h5_file, '/X', scm_feature, ...
            [local_idx, 1, 1, 1], ...
            [1, num_pairs, num_freq_bins, 2]);

        if mod(total_written_samples, max(1, floor(total_split_samples / 10))) == 0 || ...
                total_written_samples == total_split_samples
            fprintf(['  %d/%d selected windows complete. Range = %.3f km. ', ...
                'Snapshots = %d\n'], ...
                total_written_samples, total_split_samples, ...
                scm_segment_range_km(scm_idx), feature_info.num_snapshots);
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

clear signal_time_full signal_time_seg scm_freq scm_feature segments;

fprintf('SCM signal segmentation complete.\n');
