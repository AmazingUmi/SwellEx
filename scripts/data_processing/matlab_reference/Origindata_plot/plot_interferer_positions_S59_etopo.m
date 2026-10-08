% plot_interferer_positions_S59_etopo.m
% Load S59 interferer positions and overlay them on the ETOPO map.

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

%% Input path
deg_m_file = fullfile(origindata_dir, 'events', 'S59', 'Interferer_Positions_deg.m');
assert(isfile(deg_m_file) == 1, ...
    'File not found: %s. Run export_interferer_positions_S59_to_m first.', ...
    deg_m_file);

run(deg_m_file);
assert(exist('lat_deg', 'var') == 1 && exist('lon_deg', 'var') == 1, ...
    'lat_deg/lon_deg not found after running %s', deg_m_file);

lat_pts = lat_deg(:);
lon_pts = lon_deg(:);
fprintf('Loaded interferer positions: %d valid points\n', numel(lat_pts));

%% Plot limits
pad_deg = 0.10;
lat_lim = sort([min(lat_pts) - pad_deg, max(lat_pts) + pad_deg]);
lon_lim = sort([min(lon_pts) - pad_deg, max(lon_pts) + pad_deg]);

min_span_deg = 0.20;
if diff(lat_lim) < min_span_deg
    lat_center = mean(lat_lim);
    lat_lim = lat_center + [-min_span_deg / 2, min_span_deg / 2];
end
if diff(lon_lim) < min_span_deg
    lon_center = mean(lon_lim);
    lon_lim = lon_center + [-min_span_deg / 2, min_span_deg / 2];
end

fprintf('Plot range: lat=[%.6f %.6f], lon=[%.6f %.6f]\n', ...
    lat_lim(1), lat_lim(2), lon_lim(1), lon_lim(2));

%% Plot
plotgeomap(lat_lim, lon_lim);

ax = gca;
plot(ax, lon_pts, lat_pts, 'r.', 'MarkerSize', 16);

show_labels = false;
if show_labels
    for point_idx = 1:numel(lat_pts)
        text(ax, lon_pts(point_idx), lat_pts(point_idx), ...
            sprintf(' %d', point_idx), 'Color', 'r', 'FontSize', 9);
    end
end

title(ax, 'Interferer Positions on ETOPO (S59)');
legend(ax, {'Interferer'}, 'Location', 'best');
