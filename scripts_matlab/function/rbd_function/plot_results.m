function plot_results(green_time, num_elements, fs, segment_num_samples, ...
    theta_vec, beam_power, theta_best, use_plane_wave, array_depths_m, ...
    green_freq, freq_hz, theta_selected)
%PLOT_RESULTS Plot Bartlett power and Green's functions.
%
% Required inputs:
%   green_time           N x Nt estimated time-domain Green's function
%   num_elements         number of array elements
%   fs                   sampling frequency [Hz]
%   segment_num_samples  segment length [samples]
%   theta_vec            steering angle grid [rad]
%   beam_power           Bartlett beam power over theta_vec
%   theta_best           best steering angle [rad]
%   use_plane_wave       true if plane-wave delays are used
%
% Optional input:
%   array_depths_m       1 x N array element depths [m]
%   green_freq           N x Nf one-sided frequency-domain Green's function
%   freq_hz              1 x Nf one-sided frequency vector [Hz]
%   theta_selected       selected multipath steering angles [rad]

if nargin < 9
    array_depths_m = [];
end
if nargin < 10
    green_freq = [];
end
if nargin < 11
    freq_hz = [];
end
if nargin < 12
    theta_selected = theta_best;
end

has_depth_plot = ~isempty(array_depths_m);
has_freq_plot = ~isempty(green_freq) && ~isempty(freq_hz);

time_s = (0:segment_num_samples - 1) / fs;
[time_plot_ms, green_plot_db] = prepare_green_display_data( ...
    green_time, time_s, fs, segment_num_samples);
if has_freq_plot
    [freq_plot_hz, green_freq_plot_db] = prepare_green_freq_display_data( ...
        green_freq, freq_hz);
end

figure('Name', 'RBD Bartlett Beam Power', 'Position', [50 50 1200 360]);
plot_bartlett_power(theta_vec, beam_power, theta_best, use_plane_wave, ...
    theta_selected);

figure('Name', 'RBD Green Functions', 'Position', [50 50 1200 760]);
green_layout = tiledlayout(2, 2, 'TileSpacing', 'compact', ...
    'Padding', 'compact');

nexttile(green_layout, 1);
plot_green_by_element(time_plot_ms, green_plot_db, num_elements);

nexttile(green_layout, 2);
if has_freq_plot
    plot_green_freq_by_element(freq_plot_hz, green_freq_plot_db, num_elements);
else
    plot_unavailable_panel('Frequency-domain Green''s function unavailable');
end

nexttile(green_layout, 3);
if has_depth_plot
    plot_green_by_depth(time_plot_ms, green_plot_db, array_depths_m, num_elements);
else
    plot_unavailable_panel('Depth-referenced Green''s function unavailable');
end

nexttile(green_layout, 4);
if has_depth_plot && has_freq_plot
    plot_green_freq_by_depth(freq_plot_hz, green_freq_plot_db, array_depths_m, num_elements);
else
    plot_unavailable_panel(['Depth-referenced frequency-domain ', ...
        'Green''s function unavailable']);
end

title(green_layout, 'RBD Green Function Results');
end

function plot_bartlett_power(theta_vec, beam_power, theta_best, use_plane_wave, ...
    theta_selected)
theta_deg = theta_vec * 180 / pi;
beam_power_db = 10 * log10(beam_power / (max(beam_power) + eps) + eps);
if isempty(theta_selected)
    theta_selected = theta_best;
end

theta_deg_fine = linspace(theta_deg(1), theta_deg(end), ...
    max(numel(theta_deg) * 8, 1000));
beam_power_db_fine = interp1(theta_deg, beam_power_db, ...
    theta_deg_fine, 'pchip');

legend_handles = gobjects(0);
legend_labels = {};
beam_handle = plot(theta_deg_fine, beam_power_db_fine, 'b-', 'LineWidth', 1.5);
legend_handles(end + 1) = beam_handle;
legend_labels{end + 1} = 'Beam power';
hold on;
if ~isempty(theta_selected)
    theta_selected_deg = theta_selected * 180 / pi;
    arrival_colors = lines(numel(theta_selected_deg));
    for arrival_idx = 1:numel(theta_selected_deg)
        arrival_handle = xline(theta_selected_deg(arrival_idx), '--', ...
            sprintf('Arrival %d', arrival_idx), ...
            'Color', arrival_colors(arrival_idx, :), ...
            'LineWidth', 1.4, ...
            'LabelVerticalAlignment', 'middle', ...
            'LabelHorizontalAlignment', 'left');
        legend_handles(end + 1) = arrival_handle; %#ok<AGROW>
        legend_labels{end + 1} = sprintf('Arrival %d', arrival_idx); %#ok<AGROW>
    end
end
threshold_handle = yline(-3, 'k:', '-3 dB');
legend_handles(end + 1) = threshold_handle;
legend_labels{end + 1} = '-3 dB';
hold off;

xlim([theta_deg(1) theta_deg(end)]);
ylim([-20 3]);
grid on;
xlabel('Steering angle [deg]');
ylabel('Normalized power [dB]');
legend(legend_handles, legend_labels, 'Location', 'best');

if use_plane_wave
    title('Bartlett beam power with plane-wave delays');
else
    title('Bartlett beam power with ray-integral delays');
end
end

function [time_plot_ms, green_plot_db] = prepare_green_display_data( ...
    green_time, time_s, fs, segment_num_samples)
time_plot_s = time_s(1:min(segment_num_samples, round(fs * 0.1)));
green_plot_abs = abs(green_time(:, 1:numel(time_plot_s)));
green_plot_db = 20 * log10(green_plot_abs / (max(green_plot_abs(:)) + eps));
time_plot_ms = time_plot_s * 1000;
end

function [freq_plot_hz, green_freq_plot_db] = prepare_green_freq_display_data( ...
    green_freq, freq_hz)
freq_hz = freq_hz(:).';
if size(green_freq, 2) ~= numel(freq_hz)
    error('green_freq must have one column per frequency bin in freq_hz.');
end

[freq_plot_hz, green_freq_band_power] = third_octave_band_power( ...
    green_freq, freq_hz);
green_freq_plot_db = 10 * log10(green_freq_band_power / ...
    (max(green_freq_band_power(:)) + eps) + eps);
end

function plot_green_by_element(time_plot_ms, green_plot_db, num_elements)
element_idx = 1:num_elements;
time_fine_ms = linspace(time_plot_ms(1), time_plot_ms(end), ...
    max(numel(time_plot_ms) * 4, 400));
element_fine_idx = linspace(element_idx(1), element_idx(end), ...
    max(num_elements * 4, 200));

[time_grid_ms, element_grid_idx] = meshgrid(time_plot_ms, element_idx);
[time_fine_grid_ms, element_fine_grid_idx] = meshgrid(time_fine_ms, element_fine_idx);
green_plot_db_fine = interp2(time_grid_ms, element_grid_idx, green_plot_db, ...
    time_fine_grid_ms, element_fine_grid_idx, 'linear');

imagesc(time_fine_ms, element_fine_idx, green_plot_db_fine);
colormap(jet);
colorbar;
clim([-25 0]);
xlabel('Time [ms]');
ylabel('Array element j');
title('Estimated Green''s function (time domain, dB)');
end

function plot_green_freq_by_element(freq_plot_hz, green_freq_plot_db, num_elements)
element_idx = 1:num_elements;
freq_band_idx = 1:numel(freq_plot_hz);
freq_fine_idx = linspace(freq_band_idx(1), freq_band_idx(end), ...
    max(numel(freq_plot_hz) * 8, 400));
element_fine_idx = linspace(element_idx(1), element_idx(end), ...
    max(num_elements * 4, 200));

[freq_grid_idx, element_grid_idx] = meshgrid(freq_band_idx, element_idx);
[freq_fine_grid_idx, element_fine_grid_idx] = meshgrid(freq_fine_idx, ...
    element_fine_idx);
green_freq_db_fine = interp2(freq_grid_idx, element_grid_idx, ...
    green_freq_plot_db, freq_fine_grid_idx, element_fine_grid_idx, 'linear');

imagesc(freq_fine_idx, element_fine_idx, green_freq_db_fine);
colormap(jet);
colorbar;
clim([-25 0]);
apply_third_octave_frequency_ticks(freq_plot_hz);
xlabel('One-third-octave center frequency [Hz]');
ylabel('Array element j');
title('Estimated Green''s function (1/3-octave frequency-domain level, dB)');
end

function plot_green_by_depth(time_plot_ms, green_plot_db, array_depths_m, num_elements)
array_depths_m = array_depths_m(:);
[depth_sorted_m, sort_idx] = sort(array_depths_m);
green_depth_db = green_plot_db(sort_idx, :);

depth_fine_m = linspace(depth_sorted_m(1), depth_sorted_m(end), ...
    max(num_elements * 8, 300));
time_fine_ms = linspace(time_plot_ms(1), time_plot_ms(end), ...
    max(numel(time_plot_ms) * 4, 400));

[time_depth_grid_ms, depth_grid_m] = meshgrid(time_plot_ms, depth_sorted_m);
[time_fine_grid_ms, depth_fine_grid_m] = meshgrid(time_fine_ms, depth_fine_m);
green_depth_db_fine = interp2(time_depth_grid_ms, depth_grid_m, green_depth_db, ...
    time_fine_grid_ms, depth_fine_grid_m, 'linear');

imagesc(time_fine_ms, depth_fine_m, green_depth_db_fine);
set(gca, 'YDir', 'reverse');
colormap(jet);
colorbar;
clim([-25 0]);
xlabel('Time [ms]');
ylabel('Receiver depth [m]');
title('Estimated Green''s function (depth-referenced display, dB)');

hold on;
plot(time_fine_ms(1) * ones(size(depth_sorted_m)), depth_sorted_m, ...
    'ko', 'MarkerFaceColor', 'w', 'MarkerSize', 4);
hold off;
end

function plot_green_freq_by_depth(freq_plot_hz, green_freq_plot_db, ...
    array_depths_m, num_elements)
array_depths_m = array_depths_m(:);
[depth_sorted_m, sort_idx] = sort(array_depths_m);
green_freq_depth_db = green_freq_plot_db(sort_idx, :);

depth_fine_m = linspace(depth_sorted_m(1), depth_sorted_m(end), ...
    max(num_elements * 8, 300));
freq_band_idx = 1:numel(freq_plot_hz);
freq_fine_idx = linspace(freq_band_idx(1), freq_band_idx(end), ...
    max(numel(freq_plot_hz) * 8, 400));

[freq_depth_grid_idx, depth_grid_m] = meshgrid(freq_band_idx, depth_sorted_m);
[freq_fine_grid_idx, depth_fine_grid_m] = meshgrid(freq_fine_idx, depth_fine_m);
green_freq_depth_db_fine = interp2(freq_depth_grid_idx, depth_grid_m, ...
    green_freq_depth_db, freq_fine_grid_idx, depth_fine_grid_m, 'linear');

imagesc(freq_fine_idx, depth_fine_m, green_freq_depth_db_fine);
set(gca, 'YDir', 'reverse');
colormap(jet);
colorbar;
clim([-25 0]);
apply_third_octave_frequency_ticks(freq_plot_hz);
xlabel('One-third-octave center frequency [Hz]');
ylabel('Receiver depth [m]');
title('Estimated Green''s function (1/3-octave level by depth, dB)');

hold on;
plot(freq_fine_idx(1) * ones(size(depth_sorted_m)), depth_sorted_m, ...
    'ko', 'MarkerFaceColor', 'w', 'MarkerSize', 4);
hold off;
end

function plot_unavailable_panel(message_text)
axis off;
text(0.5, 0.5, message_text, ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', ...
    'FontWeight', 'bold');
end

function [center_freq_hz, band_power] = third_octave_band_power( ...
    green_freq, freq_hz)
positive_freq_idx = find(freq_hz > 0);
if isempty(positive_freq_idx)
    error('freq_hz must contain positive frequencies for 1/3-octave display.');
end

positive_freq_hz = freq_hz(positive_freq_idx);
positive_green_freq = green_freq(:, positive_freq_idx);
min_freq_hz = positive_freq_hz(1);
max_freq_hz = positive_freq_hz(end);

center_freq_candidates_hz = third_octave_center_frequencies( ...
    min_freq_hz, max_freq_hz);
num_elements = size(green_freq, 1);
band_power = zeros(num_elements, numel(center_freq_candidates_hz));
center_freq_hz = zeros(1, numel(center_freq_candidates_hz));
num_bands = 0;

for band_idx = 1:numel(center_freq_candidates_hz)
    center_freq_candidate_hz = center_freq_candidates_hz(band_idx);
    lower_edge_hz = center_freq_candidate_hz / 2^(1 / 6);
    upper_edge_hz = center_freq_candidate_hz * 2^(1 / 6);
    in_band = positive_freq_hz >= lower_edge_hz & ...
        positive_freq_hz < upper_edge_hz;

    if band_idx == numel(center_freq_candidates_hz)
        in_band = positive_freq_hz >= lower_edge_hz & ...
            positive_freq_hz <= upper_edge_hz;
    end

    if any(in_band)
        num_bands = num_bands + 1;
        center_freq_hz(num_bands) = center_freq_candidate_hz;
        band_power(:, num_bands) = mean(abs(positive_green_freq(:, in_band)).^2, 2);
    end
end

center_freq_hz = center_freq_hz(1:num_bands);
band_power = band_power(:, 1:num_bands);
end

function center_freq_hz = third_octave_center_frequencies(min_freq_hz, max_freq_hz)
reference_freq_hz = 1000;
edge_factor = 2^(1 / 6);
first_band_idx = ceil(3 * log2(min_freq_hz / edge_factor / reference_freq_hz));
last_band_idx = floor(3 * log2(max_freq_hz * edge_factor / reference_freq_hz));
band_idx = first_band_idx:last_band_idx;
center_freq_hz = reference_freq_hz * 2.^(band_idx / 3);
end

function apply_third_octave_frequency_ticks(freq_plot_hz)
num_bands = numel(freq_plot_hz);
max_ticks = 12;
tick_idx = unique(round(linspace(1, num_bands, min(num_bands, max_ticks))));

set(gca, 'XLim', [1 num_bands], ...
    'XTick', tick_idx, ...
    'XTickLabel', format_frequency_tick_labels(freq_plot_hz(tick_idx)));
end

function labels = format_frequency_tick_labels(freq_hz)
labels = cell(size(freq_hz));
for freq_idx = 1:numel(freq_hz)
    if freq_hz(freq_idx) >= 100
        labels{freq_idx} = sprintf('%.0f', freq_hz(freq_idx));
    elseif freq_hz(freq_idx) >= 10
        labels{freq_idx} = sprintf('%.1f', freq_hz(freq_idx));
    else
        labels{freq_idx} = sprintf('%.2f', freq_hz(freq_idx));
    end
end
end
