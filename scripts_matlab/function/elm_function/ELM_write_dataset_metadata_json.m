function ELM_write_dataset_metadata_json(metadata_file, split_strategy_dir_name, ...
    split_metadata, split_files, split_names, num_segments, fs, ...
    segment_duration_s, segment_step_s, segment_start_s, segment_end_s, ...
    range_file, freq_hz, array_depths_m, feature_config, dataset_variant_tag)
%ELM_WRITE_DATASET_METADATA_JSON Write ELM dataset parameters next to HDF5.
metadata = struct();
metadata.format = ['Neural-network dataset: ', ...
    'X(window, numerator_element, denominator_element, frequency, real_imag), ', ...
    'y_range_km(window).'];
metadata.real_imag_index = ...
    'X(:,:,:,:,1)=real(ratio_freq), X(:,:,:,:,2)=imag(ratio_freq).';
metadata.feature_definition = ...
    'ratio_freq(i,j,f)=FFT(element_i,f)/FFT(element_j,f), with denominator floor.';
metadata.split_strategy = split_metadata.split_strategy;
metadata.split_strategy_dir_name = split_strategy_dir_name;
metadata.dataset_variant_tag = char(dataset_variant_tag);
metadata.global_num_segments = num_segments;
metadata.candidate_segment_start_s = segment_start_s;
metadata.candidate_segment_end_s = segment_end_s;
metadata.fs_hz = fs;
metadata.segment_duration_s = segment_duration_s;
metadata.segment_step_s = segment_step_s;
metadata.range_file = range_file;
metadata.freq_num = numel(freq_hz);
metadata.freq_min_hz = min(freq_hz);
metadata.freq_max_hz = max(freq_hz);
metadata.num_elements = numel(array_depths_m);
metadata.feature_config = feature_config;
metadata.split = split_metadata;

split_info = repmat(struct('name', '', 'file', ''), 1, numel(split_files));
for split_idx = 1:numel(split_files)
    split_info(split_idx).name = split_names{split_idx};
    split_info(split_idx).file = split_files{split_idx};
end
metadata.files = split_info;

json_text = jsonencode(metadata);
fid = fopen(metadata_file, 'w');
if fid < 0
    error('Could not open metadata JSON for writing: %s', metadata_file);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', json_text);
delete(cleanup);
end
