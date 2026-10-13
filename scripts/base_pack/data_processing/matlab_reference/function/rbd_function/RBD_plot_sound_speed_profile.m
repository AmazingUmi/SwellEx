function RBD_plot_sound_speed_profile(depth_m, sound_speed_ms, ...
    sound_channel_depth_m, sound_speed_min_ms)
%RBD_PLOT_SOUND_SPEED_PROFILE Plot the sound-speed profile versus depth.

figure;
plot(sound_speed_ms, depth_m, 'b-', 'LineWidth', 1.5);
set(gca, 'YDir', 'reverse');
grid on;
xlabel('Sound speed [m/s]');
ylabel('Depth [m]');
title('Sound-speed profile');

if nargin >= 3 && ~isempty(sound_channel_depth_m)
    hold on;
    yline(sound_channel_depth_m, 'r--', 'LineWidth', 1.2);
    if nargin >= 4 && ~isempty(sound_speed_min_ms)
        label_str = sprintf('Sound channel: z_0 = %.1f m, c_0 = %.1f m/s', ...
            sound_channel_depth_m, sound_speed_min_ms);
    else
        label_str = sprintf('Sound channel: z_0 = %.1f m', sound_channel_depth_m);
    end

    text(min(sound_speed_ms) + 0.5, sound_channel_depth_m - 5, label_str, ...
        'Color', 'r', 'FontSize', 9);
    hold off;
end
end
