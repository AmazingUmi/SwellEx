function [range_time_s, range_km_raw] = DS_load_range_labels(range_file)
%DS_LOAD_RANGE_LABELS Load and sort range labels used by dataset generators.
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
end
