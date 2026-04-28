% plot_4arrays_on_etopo.m
% Plot the VLA, TLA, HLA North, and HLA South array locations on ETOPO.

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

cd(script_dir);
addpath(script_dir);
addpath(fullfile(scripts_dir, 'function'));
clear tmp;

%% Input files
arrays_file = fullfile(project_dir, 'environments', 'Arrays.mat');
etopo_file = fullfile(project_dir, 'environments', 'etopo2022_swellex.mat');

assert(isfile(arrays_file) == 1, 'File not found: %s', arrays_file);
assert(isfile(etopo_file) == 1, 'File not found: %s', etopo_file);

%% Load array metadata
loaded_arrays = load(arrays_file);
arrays = pickArraysStruct(loaded_arrays);

array_list = { ...
    struct('key', "VLA", 'label', "VLA"), ...
    struct('key', "TLA", 'label', "TLA"), ...
    struct('key', "HLA_North", 'label', "HLA North"), ...
    struct('key', "HLA_South", 'label', "HLA South") ...
    };

[array_points_deg, array_names] = extractArrayLatLon(arrays, array_list);
lat_pts = array_points_deg(:, 1);
lon_pts = array_points_deg(:, 2);

%% Global top-view plot
pad_deg = 0.15;
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

plotgeomap(lat_lim, lon_lim, false);

ax = gca;
plot(ax, lon_pts, lat_pts, 'r.', 'MarkerSize', 18);
for point_idx = 1:numel(array_names)
    text(ax, lon_pts(point_idx), lat_pts(point_idx), ...
        ['  ' char(array_names(point_idx))], 'Color', 'r', 'FontSize', 10);
end
title(ax, 'SWellEx-96 Arrays on ETOPO (Top View)');
legend(ax, {'Arrays'}, 'Location', 'best');

%% Load ETOPO for local 3D plots
etopo = load(etopo_file);
assert(isfield(etopo, 'Lon') && isfield(etopo, 'Lat') && isfield(etopo, 'Altitude'), ...
    'ETOPO file must contain Lon, Lat, and Altitude: %s', etopo_file);

lon_all = etopo.Lon;
lat_all = etopo.Lat;
elevation_m = etopo.Altitude;

upsample_factor = 4;
smooth_sigma = 1.0;
local_pad_deg = 0.02;

%% Local plots for each array
for array_idx = 1:numel(array_list)
    key = array_list{array_idx}.key;
    label = array_list{array_idx}.label;

    array_data = getArrayByKey(arrays, key);
    if ~isfield(array_data, 'element_positions') || isempty(array_data.element_positions) ...
            || height(array_data.element_positions) == 0
        fprintf('Skipping %s: no element_positions field\n', label);
        continue;
    end

    [lat_ref_deg, lon_ref_deg] = getLatLonDeg(array_data);
    element_table = array_data.element_positions;
    [north_m, east_m, depth_m, ~] = getNEDElement(element_table);
    [lat_el_deg, lon_el_deg] = ne2ll(lat_ref_deg, lon_ref_deg, north_m, east_m);

    lat_lim_local = sort([min(lat_el_deg) - local_pad_deg, max(lat_el_deg) + local_pad_deg]);
    lon_lim_local = sort([min(lon_el_deg) - local_pad_deg, max(lon_el_deg) + local_pad_deg]);

    plotgeomap(lat_lim_local, lon_lim_local, false);
    ax_local = gca;
    plot(ax_local, lon_el_deg, lat_el_deg, 'r.', 'MarkerSize', 16);
    plot(ax_local, lon_ref_deg, lat_ref_deg, 'r+', 'MarkerSize', 10, 'LineWidth', 1.5);
    title(ax_local, sprintf('%s Elements on ETOPO (Local Top View)', label));
    legend(ax_local, {'Elements', 'Ref'}, 'Location', 'best');

    if strcmpi(string(key), "VLA") || strcmpi(string(key), "TLA")
        [lon_grid_q, lat_grid_q, elev_q_m, ok_topo] = localTopoSmooth( ...
            lon_all, lat_all, elevation_m, lat_lim_local, lon_lim_local, ...
            upsample_factor, smooth_sigma);
        if ok_topo
            figure('Color', 'w');
            surf(lon_grid_q, lat_grid_q, elev_q_m, 'EdgeColor', 'none');
            hold on;
            shading interp;
            colormap(parula(256));
            cb = colorbar;
            cb.Label.String = 'Elevation (m)';
            grid on;
            xlabel('Longitude (deg)');
            ylabel('Latitude (deg)');
            zlabel('Elevation (m)');
            title(sprintf('%s Elements on ETOPO (Local 3D)', label));
            view(45, 25);

            element_z_m = -depth_m;
            plot3(lon_el_deg, lat_el_deg, element_z_m, 'r.', 'MarkerSize', 18);
            plot3(lon_ref_deg, lat_ref_deg, 0, 'r+', 'MarkerSize', 10, 'LineWidth', 1.5);
        end
    end
end

function arrays = pickArraysStruct(loaded_arrays)
if isfield(loaded_arrays, 'Arrays')
    arrays = loaded_arrays.Arrays;
    return;
end

field_names = fieldnames(loaded_arrays);
if numel(field_names) == 1
    arrays = loaded_arrays.(field_names{1});
    return;
end

error('Arrays.mat does not contain a unique Arrays structure: %s', ...
    strjoin(string(field_names), ', '));
end

function [array_points_deg, array_names] = extractArrayLatLon(arrays, array_list)
array_points_deg = nan(numel(array_list), 2);
array_names = strings(numel(array_list), 1);

for array_idx = 1:numel(array_list)
    key = array_list{array_idx}.key;
    label = array_list{array_idx}.label;
    array_names(array_idx) = label;

    array_data = getArrayByKey(arrays, key);
    [lat_deg, lon_deg] = getLatLonDeg(array_data);
    array_points_deg(array_idx, :) = [lat_deg, lon_deg];
end
end

function [lat_deg, lon_deg] = getLatLonDeg(array_data)
if isfield(array_data, 'position_ll_deg') ...
        && isfield(array_data.position_ll_deg, 'lat') ...
        && isfield(array_data.position_ll_deg, 'lon')
    lat_deg = double(array_data.position_ll_deg.lat);
    lon_deg = double(array_data.position_ll_deg.lon);
    return;
end

if isfield(array_data, 'lat') && isfield(array_data, 'lon')
    lat_deg = double(array_data.lat);
    lon_deg = double(array_data.lon);
    return;
end

if isfield(array_data, 'Lat') && isfield(array_data, 'Lon')
    lat_deg = double(array_data.Lat);
    lon_deg = double(array_data.Lon);
    return;
end

error('Array metadata does not contain usable latitude/longitude fields.');
end

function array_data = getArrayByKey(arrays, key)
if isfield(arrays, key)
    array_data = arrays.(key);
    return;
end

field_names = fieldnames(arrays);
match_name = field_names(strcmpi(field_names, key));
assert(~isempty(match_name), 'Array key not found: %s', key);
array_data = arrays.(match_name{1});
end

function [north_m, east_m, depth_m, element_idx] = getNEDElement(element_table)
var_names_lower = lower(string(element_table.Properties.VariableNames));
element_idx = pickCol(element_table, var_names_lower, ["element", "elem"]);
north_m = pickCol(element_table, var_names_lower, ["north", "n"]);
east_m = pickCol(element_table, var_names_lower, ["east", "e", "x"]);
depth_m = pickCol(element_table, var_names_lower, ["depth", "z"]);

if isempty(element_idx)
    element_idx = (1:height(element_table)).';
end
if isempty(north_m)
    north_m = zeros(height(element_table), 1);
end
if isempty(east_m)
    east_m = zeros(height(element_table), 1);
end
if isempty(depth_m)
    depth_m = nan(height(element_table), 1);
end
end

function col = pickCol(element_table, var_names_lower, key_list)
col = [];
for key_idx = 1:numel(key_list)
    match_idx = find(var_names_lower == key_list(key_idx), 1);
    if ~isempty(match_idx)
        col = double(element_table.(element_table.Properties.VariableNames{match_idx}));
        return;
    end
end
end

function [lat_deg, lon_deg] = ne2ll(lat_ref_deg, lon_ref_deg, north_m, east_m)
earth_radius_m = 6371000;
delta_lat_deg = (north_m ./ earth_radius_m) * (180 / pi);
delta_lon_deg = (east_m ./ (earth_radius_m * cosd(lat_ref_deg))) * (180 / pi);
lat_deg = lat_ref_deg + delta_lat_deg;
lon_deg = lon_ref_deg + delta_lon_deg;
end

function [lon_grid_q, lat_grid_q, elev_q_m, ok] = localTopoSmooth( ...
    lon_all, lat_all, elevation_m, lat_lim, lon_lim, upsample_factor, smooth_sigma)
ok = false;
lon_grid_q = [];
lat_grid_q = [];
elev_q_m = [];

lon_idx = lon_all >= lon_lim(1) & lon_all <= lon_lim(2);
lat_idx = lat_all >= lat_lim(1) & lat_all <= lat_lim(2);
if ~any(lon_idx) || ~any(lat_idx)
    return;
end

lon_vec = lon_all(lon_idx);
lat_vec = lat_all(lat_idx);
elev_grid_m = double(elevation_m(lon_idx, lat_idx)).';

lon_query = linspace(lon_vec(1), lon_vec(end), max(2, numel(lon_vec) * upsample_factor));
lat_query = linspace(lat_vec(1), lat_vec(end), max(2, numel(lat_vec) * upsample_factor));
[lon_grid_q, lat_grid_q] = meshgrid(lon_query, lat_query);

elev_q_m = interp2(lon_vec, lat_vec, elev_grid_m, lon_grid_q, lat_grid_q, 'spline');
if exist('imgaussfilt', 'file') == 2
    elev_q_m = imgaussfilt(elev_q_m, smooth_sigma, 'Padding', 'replicate');
else
    kernel_radius = max(1, ceil(3 * smooth_sigma));
    kernel_x = -kernel_radius:kernel_radius;
    kernel_g = exp(-(kernel_x.^2) / (2 * smooth_sigma^2));
    kernel_g = kernel_g / sum(kernel_g);
    elev_q_m = conv2(conv2(elev_q_m, kernel_g, 'same'), kernel_g', 'same');
end

ok = true;
end
