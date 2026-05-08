function [split_indices, split_names, segment_split_idx, split_metadata] = ...
    SS_build_split_indices(split_strategy, num_segments, valid_sample, ...
    segment_range_km, segment_center_time_s, segment_step_s, ...
    split_options)
%SS_BUILD_SPLIT_INDICES Build train/test segment indices for a split strategy.
split_strategy = string(split_strategy);
split_names = {'train', 'test'};
segment_split_idx = zeros(1, num_segments, 'uint8');

switch split_strategy
    case "periodic"
        [train_segment_idx, test_segment_idx, split_metadata] = ...
            build_periodic_split(num_segments, split_options);
    case "Range_nearby"
        [train_segment_idx, test_segment_idx, split_metadata] = ...
            build_range_nearby_split(num_segments, valid_sample, ...
            segment_range_km, segment_center_time_s, segment_step_s, ...
            split_options);
    otherwise
        error('Unsupported split_strategy: %s.', split_strategy);
end

split_metadata.split_strategy = char(split_strategy);
split_metadata.num_train_samples = numel(train_segment_idx);
split_metadata.num_test_samples = numel(test_segment_idx);
split_metadata.train_segment_idx_first = first_or_nan(train_segment_idx);
split_metadata.train_segment_idx_last = last_or_nan(train_segment_idx);
split_metadata.test_segment_idx_first = first_or_nan(test_segment_idx);
split_metadata.test_segment_idx_last = last_or_nan(test_segment_idx);
split_metadata.sample_indices_dataset = '/split/source_segment_idx';

if isempty(train_segment_idx)
    error('Split strategy %s produced an empty training split.', split_strategy);
end
if isempty(test_segment_idx)
    error('Split strategy %s produced an empty test split.', split_strategy);
end

split_indices = {train_segment_idx, test_segment_idx};
segment_split_idx(train_segment_idx) = 1;
segment_split_idx(test_segment_idx) = 2;
end

function [train_segment_idx, test_segment_idx, split_metadata] = ...
    build_periodic_split(num_segments, split_options)
require_option(split_options, 'train_test_ratio', 'periodic');
train_test_ratio = split_options.train_test_ratio;
if numel(train_test_ratio) ~= 2 || any(train_test_ratio < 1) || ...
        any(train_test_ratio ~= round(train_test_ratio))
    error('train_test_ratio must contain two positive integers, e.g. [4 1].');
end

split_period = sum(train_test_ratio);
split_position = mod(0:num_segments - 1, split_period) + 1;
test_mask = split_position > train_test_ratio(1);
if ~any(test_mask) && num_segments > 1
    test_mask(end) = true;
end

train_segment_idx = find(~test_mask);
test_segment_idx = find(test_mask);

split_metadata = struct();
split_metadata.train_test_ratio = train_test_ratio;
split_metadata.split_period = split_period;
end

function [train_segment_idx, test_segment_idx, split_metadata] = ...
    build_range_nearby_split(num_segments, valid_sample, segment_range_km, ...
    segment_center_time_s, segment_step_s, split_options)
require_option(split_options, 'half_duration_s', 'Range_nearby');
require_option(split_options, 'gap_s', 'Range_nearby');
require_option(split_options, 'train_side', 'Range_nearby');

range_nearby_half_duration_s = split_options.half_duration_s;
range_nearby_gap_s = split_options.gap_s;
range_nearby_train_side = split_options.train_side;

if range_nearby_half_duration_s <= 0
    error('range_nearby_half_duration_s must be positive.');
end
if range_nearby_gap_s < 0
    error('range_nearby_gap_s must be non-negative.');
end
if segment_step_s <= 0
    error('segment_step_s must be positive.');
end

valid_idx = find(valid_sample & isfinite(segment_range_km));
if isempty(valid_idx)
    error('Range_nearby split requires at least one finite range label.');
end

[range_min_km, min_local_idx] = min(segment_range_km(valid_idx));
range_min_segment_idx = valid_idx(min_local_idx);
range_nearby_half_num_segments = ...
    max(1, floor(range_nearby_half_duration_s / segment_step_s));
range_nearby_gap_num_segments = ...
    floor(range_nearby_gap_s / segment_step_s);

max_left = range_min_segment_idx - range_nearby_gap_num_segments - 1;
max_right = num_segments - range_min_segment_idx - range_nearby_gap_num_segments;
usable_num_segments = min([range_nearby_half_num_segments, max_left, max_right]);
if usable_num_segments < 1
    error(['Range_nearby split cannot build symmetric windows around segment %d. ', ...
        'Reduce range_nearby_gap_s or choose a wider segmentation interval.'], ...
        range_min_segment_idx);
end

left_idx = range_min_segment_idx - range_nearby_gap_num_segments - ...
    usable_num_segments : range_min_segment_idx - ...
    range_nearby_gap_num_segments - 1;
right_idx = range_min_segment_idx + range_nearby_gap_num_segments + 1 : ...
    range_min_segment_idx + range_nearby_gap_num_segments + usable_num_segments;

pair_valid = valid_sample(left_idx) & valid_sample(right_idx) & ...
    isfinite(segment_range_km(left_idx)) & isfinite(segment_range_km(right_idx));
left_idx = left_idx(pair_valid);
right_idx = right_idx(pair_valid);
if isempty(left_idx)
    error('Range_nearby split has no valid symmetric train/test pairs.');
end

switch string(range_nearby_train_side)
    case "before"
        train_segment_idx = left_idx;
        test_segment_idx = right_idx;
    case "after"
        train_segment_idx = right_idx;
        test_segment_idx = left_idx;
    otherwise
        error('range_nearby_train_side must be "before" or "after".');
end

split_metadata = struct();
split_metadata.range_nearby_train_side = char(range_nearby_train_side);
split_metadata.range_nearby_half_duration_s = range_nearby_half_duration_s;
split_metadata.range_nearby_gap_s = range_nearby_gap_s;
split_metadata.range_nearby_half_num_segments = range_nearby_half_num_segments;
split_metadata.range_nearby_gap_num_segments = range_nearby_gap_num_segments;
split_metadata.range_nearby_usable_num_segments_per_side = numel(left_idx);
split_metadata.range_min_segment_idx = range_min_segment_idx;
split_metadata.range_min_time_s = segment_center_time_s(range_min_segment_idx);
split_metadata.range_min_km = range_min_km;
split_metadata.left_segment_idx_first = first_or_nan(left_idx);
split_metadata.left_segment_idx_last = last_or_nan(left_idx);
split_metadata.right_segment_idx_first = first_or_nan(right_idx);
split_metadata.right_segment_idx_last = last_or_nan(right_idx);
split_metadata.left_time_start_s = segment_center_time_s(left_idx(1));
split_metadata.left_time_end_s = segment_center_time_s(left_idx(end));
split_metadata.right_time_start_s = segment_center_time_s(right_idx(1));
split_metadata.right_time_end_s = segment_center_time_s(right_idx(end));
split_metadata.left_range_km_min = min(segment_range_km(left_idx));
split_metadata.left_range_km_max = max(segment_range_km(left_idx));
split_metadata.right_range_km_min = min(segment_range_km(right_idx));
split_metadata.right_range_km_max = max(segment_range_km(right_idx));
end

function value = first_or_nan(values)
if isempty(values)
    value = NaN;
else
    value = values(1);
end
end

function value = last_or_nan(values)
if isempty(values)
    value = NaN;
else
    value = values(end);
end
end

function require_option(split_options, option_name, split_strategy)
if ~isfield(split_options, option_name)
    error('%s split requires split_options.%s.', split_strategy, option_name);
end
end
