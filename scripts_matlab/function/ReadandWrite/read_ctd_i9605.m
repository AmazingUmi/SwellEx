% read_ctd_i9605.m
% Read SWellEx-96 CTD profile file i9605.prn and save the parsed result.

%% Environment setup
clear; close all; clc;

try
    tmp = matlab.desktop.editor.getActive;
    script_dir = fileparts(tmp.Filename);
catch
    script_dir = fileparts(mfilename('fullpath'));
end

function_dir = fileparts(script_dir);
scripts_dir = fileparts(function_dir);
project_dir = fileparts(scripts_dir);
origindata_dir = fullfile(project_dir, 'Origindata');

cd(script_dir);
addpath(script_dir);
addpath(genpath(function_dir));
clear tmp;

%% Input and output paths
ctd_file = fullfile(origindata_dir, 'environments', 'ctds', 'i9605.prn');
assert(isfile(ctd_file) == 1, 'File not found: %s', ctd_file);

output_file = fullfile(origindata_dir, 'events', 'S5', 'CTD_i9605.mat');
output_dir = fileparts(output_file);
if ~isfolder(output_dir)
    mkdir(output_dir);
end

%% Read CTD file
[ctd_table, ctd_info] = read_ctd_prn(ctd_file);
disp(ctd_table(1:min(10, height(ctd_table)), :));

%% Plot
figure('Color', 'w');
tiledlayout(1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
plot(ctd_table.sound_speed_ms, ctd_table.depth_m, 'k-');
set(gca, 'YDir', 'reverse');
grid on;
xlabel('Sound speed (m/s)');
ylabel('Depth (m)');
title('c(z)');

nexttile;
plot(ctd_table.temperature_C, ctd_table.depth_m, 'r-');
set(gca, 'YDir', 'reverse');
grid on;
xlabel('Temperature (deg C)');
ylabel('Depth (m)');
title('T(z)');

nexttile;
plot(ctd_table.salinity_PSU, ctd_table.depth_m, 'b-');
set(gca, 'YDir', 'reverse');
grid on;
xlabel('Salinity (PSU)');
ylabel('Depth (m)');
title('S(z)');

sgtitle(sprintf('CTD: %s', ctd_info.filename));

%% Save
save_output = true;
if save_output
    T = ctd_table;
    CTD = ctd_info;
    save(output_file, 'T', 'CTD');
    fprintf('Saved: %s\n', output_file);
end

%% Local function
function [ctd_table, ctd_info] = read_ctd_prn(file_path)
data = readmatrix(file_path, 'FileType', 'text');
if isempty(data)
    error('No valid numeric data found in: %s', file_path);
end

if size(data, 2) < 5
    error('CTD file must contain at least 5 columns: %s', file_path);
end

data = data(:, 1:5);
valid_rows = all(isfinite(data), 2);
data = data(valid_rows, :);

depth_m = data(:, 1);
temperature_C = data(:, 2);
salinity_PSU = data(:, 3);
sound_speed_ms = data(:, 4);
sigma_t = data(:, 5);

ctd_table = table(depth_m, temperature_C, salinity_PSU, sound_speed_ms, sigma_t);

ctd_info = struct();
[~, file_name, file_ext] = fileparts(file_path);
ctd_info.filename = [file_name, file_ext];
ctd_info.path = file_path;
ctd_info.n = height(ctd_table);
ctd_info.depth_min_m = min(depth_m);
ctd_info.depth_max_m = max(depth_m);
end
