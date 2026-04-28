function plot_results_series(result)
%PLOT_RESULTS_SERIES Plot sliding-window RBD time-series results.
%
% Supported result fields:
%   window_center_time_s or t_window_center
%   theta_vec
%   theta_best_history
%   beam_power_history or B_power_history
%   green_delay_s or green_delay
%   green_selected
%   selected_element_idx or selected_elements
%   green_peak_amp
%   green_peak_delay_s or green_peak_delay
%   num_elements or N
%   use_plane_wave

window_center_time_s = get_result_field(result, ...
    {'window_center_time_s', 't_window_center'});
theta_vec = result.theta_vec;
theta_best_history = result.theta_best_history;
beam_power_history = get_result_field(result, ...
    {'beam_power_history', 'B_power_history'});
green_delay_s = get_result_field(result, {'green_delay_s', 'green_delay'});
green_selected = result.green_selected;
selected_element_idx = get_result_field(result, ...
    {'selected_element_idx', 'selected_elements'});
green_peak_amp = result.green_peak_amp;
green_peak_delay_s = get_result_field(result, ...
    {'green_peak_delay_s', 'green_peak_delay'});
num_elements = get_result_field(result, {'num_elements', 'N'});
use_plane_wave = result.use_plane_wave;

plot_best_angle(window_center_time_s, theta_best_history, use_plane_wave);
plot_bartlett_angle_time(window_center_time_s, theta_vec, ...
    theta_best_history, beam_power_history);
plot_green_selected(window_center_time_s, green_delay_s, ...
    green_selected, selected_element_idx);
plot_green_peak_evolution(window_center_time_s, green_peak_amp, ...
    green_peak_delay_s, num_elements);
end

function value = get_result_field(result, field_names)
for field_idx = 1:numel(field_names)
    if isfield(result, field_names{field_idx})
        value = result.(field_names{field_idx});
        return;
    end
end

error('Missing expected result field. Tried: %s', strjoin(field_names, ', '));
end

function plot_best_angle(window_center_time_s, theta_best_history, use_plane_wave)
figure('Position', [100 100 1100 420]);
plot(window_center_time_s, theta_best_history * 180 / pi, ...
    'k-', 'LineWidth', 1.6);
grid on;
xlabel('Analysis time [s]');
ylabel('Best steering angle [deg]');

if use_plane_wave
    title('Best Bartlett steering angle versus time (plane wave)');
else
    title('Best Bartlett steering angle versus time (ray-integral)');
end
end

function plot_bartlett_angle_time(window_center_time_s, theta_vec, ...
    theta_best_history, beam_power_history)
theta_deg = theta_vec * 180 / pi;
beam_power_db = normalize_power_history(beam_power_history);

figure('Position', [120 120 1100 520]);
imagesc(window_center_time_s, theta_deg, beam_power_db);
axis xy;
colormap(jet);
colorbar;
clim([-20 0]);
xlabel('Analysis time [s]');
ylabel('Steering angle [deg]');
title('Normalized Bartlett power versus steering angle and time');

hold on;
plot(window_center_time_s, theta_best_history * 180 / pi, ...
    'w-', 'LineWidth', 1.8);
hold off;
end

function beam_power_db = normalize_power_history(beam_power_history)
beam_power_db = 10 * log10(beam_power_history ./ ...
    (max(beam_power_history, [], 1) + eps));
end

function plot_green_selected(window_center_time_s, green_delay_s, ...
    green_selected, selected_element_idx)
green_selected_db = normalize_green_selected(green_selected);
green_delay_ms = green_delay_s * 1000;
num_selected = numel(selected_element_idx);

figure('Position', [140 80 1200 760]);
tiledlayout(num_selected, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

for selected_idx = 1:num_selected
    nexttile;
    green_map = squeeze(green_selected_db(selected_idx, :, :));

    imagesc(window_center_time_s, green_delay_ms, green_map);
    axis xy;
    colormap(jet);
    clim([-25 0]);
    ylabel(sprintf('Elem %d\nDelay [ms]', selected_element_idx(selected_idx)));

    if selected_idx == 1
        title('Estimated time-domain Green''s function versus analysis time');
    end

    if selected_idx == num_selected
        xlabel('Analysis time [s]');
    else
        set(gca, 'XTickLabel', []);
    end
end

cb = colorbar;
cb.Layout.Tile = 'east';
cb.Label.String = 'Relative amplitude [dB]';
end

function green_selected_db = normalize_green_selected(green_selected)
green_selected_abs = abs(green_selected);
green_selected_db = 20 * log10(green_selected_abs / ...
    (max(green_selected_abs(:)) + eps));
end

function plot_green_peak_evolution(window_center_time_s, green_peak_amp, ...
    green_peak_delay_s, num_elements)
green_peak_amp_db = 20 * log10(green_peak_amp ./ ...
    (max(green_peak_amp(:)) + eps));

figure('Position', [160 100 1200 720]);
tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
imagesc(window_center_time_s, 1:num_elements, green_peak_amp_db);
axis xy;
colormap(jet);
colorbar;
clim([-25 0]);
xlabel('Analysis time [s]');
ylabel('Array element j');
title('Green''s function peak amplitude versus element and time');

nexttile;
imagesc(window_center_time_s, 1:num_elements, green_peak_delay_s * 1000);
axis xy;
colormap(jet);
colorbar;
xlabel('Analysis time [s]');
ylabel('Array element j');
title('Green''s function peak delay versus element and time [ms]');
end
