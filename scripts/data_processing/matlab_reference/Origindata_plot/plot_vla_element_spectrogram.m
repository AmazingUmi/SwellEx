% plot_vla_element_spectrogram.m
% Load one VLA element time series and plot its spectrogram.
%
% Default input:
%   Origindata/events/S5/vla_matfiles/S5_VLA_NO_<element_index>.mat

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
origindata_dir = fullfile(project_dir, 'Origindata');

cd(script_dir);
addpath(script_dir);
addpath(genpath(fullfile(scripts_dir, 'function')));
clear tmp;

%% User options
event_name = 'S5';
fs = 1500;                         % sampling frequency [Hz]
element_index = 1;                 % VLA element/channel index in mat file name

time_range_s = [0 4500];            % [] for full record, or [start_s stop_s]
freq_range_hz = [49 401];           % [] for full one-sided frequency range

window_length_s = 1.0;             % STFT window length [s]
window_overlap_ratio = 0.5;       % 0 to <1
nfft = [];                         % [] uses nextpow2(window_num_samples)

remove_mean = true;
normalize_by_std = false;
dynamic_range_db = 80;             % color lower limit relative to local max

save_figure = false;
figure_ext = 'png';                % 'png', 'fig', 'pdf', ...

%% Input paths and metadata
event_dir = fullfile(origindata_dir, 'events', event_name);
vla_data_dir = fullfile(event_dir, 'vla_matfiles');
assert(isfolder(vla_data_dir) == 1, 'VLA data directory not found: %s', vla_data_dir);

element_file = sprintf('%s_VLA_NO_%d.mat', event_name, element_index);
element_path = fullfile(vla_data_dir, element_file);
assert(isfile(element_path) == 1, 'VLA element file not found: %s', element_path);

position_file = fullfile(origindata_dir, 'positions', 'positions_vla.txt');
array_depth_m = NaN;
if isfile(position_file)
    position_table = readmatrix(position_file);
    row_idx = find(position_table(:, 1) == element_index, 1, 'first');
    if ~isempty(row_idx)
        array_depth_m = position_table(row_idx, 2);
    end
end

%% Load element data
data = load(element_path);
assert(isfield(data, 'x') == 1, 'Expected variable x in %s.', element_path);

signal = double(data.x(:));
num_samples = numel(signal);
record_duration_s = num_samples / fs;

if isempty(time_range_s)
    sample_start_idx = 1;
    sample_stop_idx = num_samples;
else
    assert(numel(time_range_s) == 2 && time_range_s(2) > time_range_s(1), ...
        'time_range_s must be [] or [start_s stop_s].');
    sample_start_idx = max(1, floor(time_range_s(1) * fs) + 1);
    sample_stop_idx = min(num_samples, ceil(time_range_s(2) * fs));
end
assert(sample_start_idx < sample_stop_idx, ...
    'Selected time range is empty. Record duration is %.3f s.', record_duration_s);

signal = signal(sample_start_idx:sample_stop_idx);
selected_start_s = (sample_start_idx - 1) / fs;
time_offset_s = selected_start_s;

if remove_mean
    signal = signal - mean(signal, 'omitnan');
end
if normalize_by_std
    signal_std = std(signal, 0, 'omitnan');
    if signal_std > 0
        signal = signal / signal_std;
    end
end

%% Spectrogram
window_num_samples = max(1, round(window_length_s * fs));
assert(numel(signal) >= window_num_samples, ...
    'Selected signal length %.3f s is shorter than window_length_s %.3f s.', ...
    numel(signal) / fs, window_length_s);
assert(window_overlap_ratio >= 0 && window_overlap_ratio < 1, ...
    'window_overlap_ratio must be in [0, 1).');
overlap_num_samples = round(window_num_samples * window_overlap_ratio);
overlap_num_samples = min(overlap_num_samples, window_num_samples - 1);

if isempty(nfft)
    nfft = 2 ^ nextpow2(window_num_samples);
end
nfft = max(nfft, window_num_samples);

[stft_complex, freq_hz, time_s] = spectrogram( ...
    signal, hamming(window_num_samples, 'periodic'), overlap_num_samples, nfft, fs);

power_db = 20 * log10(abs(stft_complex) + eps);
time_s = time_s + time_offset_s;

if ~isempty(freq_range_hz)
    freq_mask = freq_hz >= freq_range_hz(1) & freq_hz <= freq_range_hz(2);
    freq_hz = freq_hz(freq_mask);
    power_db = power_db(freq_mask, :);
end

clim_max = max(power_db(:));
clim_min = clim_max - dynamic_range_db;

%% Plot
fig = figure('Color', 'w', 'Name', sprintf('%s VLA %d Spectrogram', event_name, element_index));
ax = axes(fig);

imagesc(ax, time_s / 60, freq_hz, power_db);
axis(ax, 'xy');
grid(ax, 'on');
box(ax, 'on');
try
    colormap(ax, turbo);
catch
    colormap(ax, parula);
end
cb = colorbar(ax);
cb.Label.String = 'Magnitude (dB)';
caxis(ax, [clim_min clim_max]);

xlabel(ax, 'Time Since Event Start (min)');
ylabel(ax, 'Frequency (Hz)');
title_text = sprintf('SWellEx-96 Event %s VLA Element %d', event_name, element_index);
if isfinite(array_depth_m)
    title_text = sprintf('%s, Depth %.1f m', title_text, array_depth_m);
end
title(ax, title_text);

fprintf('Loaded %s\n', element_path);
fprintf('Record duration: %.2f s; plotted %.2f to %.2f s\n', ...
    record_duration_s, time_s(1), time_s(end));
fprintf('Spectrogram: window %.3f s, overlap %.0f%%, nfft %d\n', ...
    window_num_samples / fs, 100 * overlap_num_samples / window_num_samples, nfft);

%% Save figure
if save_figure
    output_name = sprintf('%s_VLA_NO_%d_spectrogram_%0.0f_%0.0fs.%s', ...
        event_name, element_index, time_s(1), time_s(end), figure_ext);
    output_path = fullfile(script_dir, output_name);
    writeFigure(fig, output_path, figure_ext);
    fprintf('Figure saved to %s\n', output_path);
end

%% Local functions
function writeFigure(fig, output_path, figure_ext)
switch lower(figure_ext)
    case 'fig'
        savefig(fig, output_path);
    otherwise
        try
            exportgraphics(fig, output_path, 'Resolution', 300);
        catch
            saveas(fig, output_path);
        end
end
end
