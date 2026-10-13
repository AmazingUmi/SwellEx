function [signal_time_full, array_depths_m, record_duration_s] = ...
    DS_load_vla_signals(project_dir, event_name, fs)
%DS_LOAD_VLA_SIGNALS Load centered VLA element time series and array depths.
origindata_dir = DS_get_origindata_dir(project_dir);
position_file = fullfile(origindata_dir, 'positions', 'positions_vla.txt');
position_table = readmatrix(position_file);   % [channel_index, depth_m]
array_depths_m = flip(position_table(:, 2).');
num_elements = numel(array_depths_m);

event_dir = fullfile(origindata_dir, 'events', event_name);
channel_data_dir = fullfile(event_dir, "vla_matfiles");

fprintf('Loading VLA time series...\n');

first_channel_file = sprintf('%s_VLA_NO_%d.mat', event_name, 1);
first_channel_data = load(fullfile(channel_data_dir, first_channel_file));
first_channel_signal = first_channel_data.x(:).';

signal_time_full = zeros(num_elements, numel(first_channel_signal));
signal_time_full(1, :) = first_channel_signal;

for channel_idx = 2:num_elements
    channel_file = sprintf('%s_VLA_NO_%d.mat', event_name, channel_idx);
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
end
