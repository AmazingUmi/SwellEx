function segments = DS_build_segments(signal_time_full, fs, ...
    segment_duration_s, segment_step_s, segment_start_s, segment_end_s, ...
    range_time_s, range_km_raw)
%DS_BUILD_SEGMENTS Build fixed-duration segment metadata and range labels.
num_elements = size(signal_time_full, 1);
record_num_samples = size(signal_time_full, 2);
record_duration_s = record_num_samples / fs;
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

segments = struct();
segments.segment_duration_s = segment_duration_s;
segments.segment_step_s = segment_step_s;
segments.segment_start_s = segment_start_s;
segments.segment_end_s = segment_end_s;
segments.segment_num_samples = segment_num_samples;
segments.record_duration_s = record_duration_s;
segments.segment_start_time_s = segment_start_time_s;
segments.segment_center_time_s = segment_center_time_s;
segments.segment_stop_time_s = segment_stop_time_s;
segments.segment_range_km = segment_range_km;
segments.valid_sample = valid_sample;
segments.segment_sample_start_idx = segment_sample_start_idx;
segments.segment_sample_stop_idx = segment_sample_stop_idx;
segments.num_segments = num_segments;
end
