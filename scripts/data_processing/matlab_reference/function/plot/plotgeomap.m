function plotgeomap(lat_lim, lon_lim, show_mesh)
%PLOTGEOMAP Plot local ETOPO bathymetry for the requested latitude/longitude range.

if nargin < 3
    show_mesh = true;
end

this_file = mfilename('fullpath');
plot_dir = fileparts(this_file);
function_dir = fileparts(plot_dir);
scripts_dir = fileparts(function_dir);
project_dir = fileparts(scripts_dir);
origindata_dir = fullfile(project_dir, 'Origindata');

etopo_file = fullfile(origindata_dir, 'environments', 'etopo2022_swellex.mat');
etopo = load(etopo_file);

lon_all = etopo.Lon;
lat_all = etopo.Lat;
depth_all = etopo.Altitude;

lon_idx = lon_all >= lon_lim(1) & lon_all <= lon_lim(end);
lat_idx = lat_all >= lat_lim(1) & lat_all <= lat_lim(end);
depth_crop = depth_all(lon_idx, lat_idx);

lon_vec = lon_all(lon_idx);
lat_vec = lat_all(lat_idx);
depth_grid = depth_crop.';   % lat x lon

upsample_factor = 4;
smooth_sigma = 1.0;

lon_query = linspace(lon_vec(1), lon_vec(end), ...
    max(2, numel(lon_vec) * upsample_factor));
lat_query = linspace(lat_vec(1), lat_vec(end), ...
    max(2, numel(lat_vec) * upsample_factor));
[lon_grid_q, lat_grid_q] = meshgrid(lon_query, lat_query);

depth_query = interp2(lon_vec, lat_vec, depth_grid, lon_grid_q, lat_grid_q, 'spline');

if exist('imgaussfilt', 'file') == 2
    depth_query = imgaussfilt(depth_query, smooth_sigma, 'Padding', 'replicate');
else
    kernel_radius = max(1, ceil(3 * smooth_sigma));
    kernel_x = -kernel_radius:kernel_radius;
    kernel_g = exp(-(kernel_x.^2) / (2 * smooth_sigma^2));
    kernel_g = kernel_g / sum(kernel_g);
    depth_query = conv2(conv2(depth_query, kernel_g, 'same'), kernel_g', 'same');
end

if show_mesh
    figure;
    mesh(lon_query, lat_query, depth_query);
    xlabel('Longitude [deg]');
    ylabel('Latitude [deg]');
    colorbar;
    apply_bathy_colormap(depth_query);
end

figure;
imagesc(lon_query, lat_query, depth_query);
axis xy;
axis equal;
xlabel('Longitude [deg]');
ylabel('Latitude [deg]');
colorbar;
apply_bathy_colormap(depth_query);
hold on;
end

function apply_bathy_colormap(depth_query)
depth_max = max(depth_query(:));
depth_min = min(depth_query(:));

if depth_max > 0 && depth_min < 0
    num_water = 1000;
    num_land = round(num_water * abs(depth_max / depth_min));
    water_map = [linspace(0, 0.1, num_water); ...
                 linspace(0, 0.6, num_water); ...
                 linspace(0.5046, 0.8, num_water)].';
    colormap(cat(1, water_map, summer(num_land)));
end
end
