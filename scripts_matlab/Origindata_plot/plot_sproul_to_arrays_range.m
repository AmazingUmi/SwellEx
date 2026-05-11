% plot_sproul_to_arrays_range.m
% Plot range from R/V Sproul to each array versus time for a SWellEx event.
%
% Default input:
%   Origindata/events/range/RangeEventS5/SproulTo*.S5.txt

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
save_figure = true;
figure_ext = 'png';   % 'png', 'fig', 'pdf', ...

%% Input paths
range_dir = fullfile(origindata_dir, 'events', 'range', ['RangeEvent' event_name]);
assert(isfolder(range_dir) == 1, 'Range directory not found: %s', range_dir);

series_list = { ...
    struct('name', 'VLA',       'file', sprintf('SproulToVLA.%s.txt', event_name),  'marker', '+', 'line', '-'), ...
    struct('name', 'TLA',       'file', sprintf('SproulToTLA.%s.txt', event_name),  'marker', 'x', 'line', '-'), ...
    struct('name', 'HLA North', 'file', sprintf('SproulToHLAN.%s.txt', event_name), 'marker', '^', 'line', '-'), ...
    struct('name', 'HLA South', 'file', sprintf('SproulToHLAS.%s.txt', event_name), 'marker', 'v', 'line', '-') ...
    };

%% Load data
num_series = numel(series_list);
range_series = cell(num_series, 1);

for idx = 1:num_series
    file_path = fullfile(range_dir, series_list{idx}.file);
    assert(isfile(file_path) == 1, 'File not found: %s', file_path);
    range_series{idx} = readSproulRangeFile(file_path, series_list{idx}.name);
end

base_duration = range_series{1}.Duration_min;
for idx = 2:num_series
    assert(isequal(base_duration, range_series{idx}.Duration_min), ...
        'Duration axis mismatch between %s and %s.', ...
        series_list{1}.name, series_list{idx}.name);
end

%% Plot
fig = figure('Color', 'w', 'Name', ['Sproul Range ' event_name]);
ax = axes(fig);
hold(ax, 'on');

color_order = lines(num_series);
marker_step = max(1, round(numel(base_duration) / 15));
marker_index = 1:marker_step:numel(base_duration);
if marker_index(end) ~= numel(base_duration)
    marker_index(end + 1) = numel(base_duration);
end

for idx = 1:num_series
    plot(ax, ...
        range_series{idx}.Duration_min, ...
        range_series{idx}.Range_km, ...
        'LineStyle', series_list{idx}.line, ...
        'LineWidth', 1.5, ...
        'Color', color_order(idx, :), ...
        'Marker', series_list{idx}.marker, ...
        'MarkerIndices', marker_index, ...
        'MarkerSize', 7, ...
        'DisplayName', series_list{idx}.name);
end

grid(ax, 'on');
box(ax, 'on');
xlabel(ax, 'Time Since Event Start (min)');
ylabel(ax, 'Range from Sproul to Array (km)');
title(ax, sprintf('SWellEx-96 Event %s Range to Each Array', event_name));
legend(ax, 'Location', 'best');
xlim(ax, [min(base_duration), max(base_duration)]);

fprintf('Loaded %d time samples from %s\n', numel(base_duration), range_dir);

%% Save figure
if save_figure
    output_name = sprintf('Sproul_to_arrays_range_%s.%s', event_name, figure_ext);
    output_path = fullfile(script_dir, output_name);
    writeFigure(fig, output_path, figure_ext);
    fprintf('Figure saved to %s\n', output_path);
end

%% Local functions
function T = readSproulRangeFile(file_path, series_name)
fid = fopen(file_path, 'r');
assert(fid ~= -1, 'Unable to open file: %s', file_path);

cleanup = onCleanup(@() fclose(fid));
data = textscan(fid, '%f %s %f %f', 'HeaderLines', 1, 'MultipleDelimsAsOne', true);

num_rows = numel(data{1});
assert(num_rows > 0, 'No data rows found in %s', file_path);

T = table();
T.Jday = data{1};
T.Time = string(data{2});
T.Duration_min = data{3};
T.Range_km = data{4};
T.Properties.Description = series_name;
end

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
