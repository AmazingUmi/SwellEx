function tau_matrix = RBD_compute_tau(theta_vec, array_depths_m, sound_speed_ms, ...
    sound_speed_depth_m, use_plane_wave)
%RBD_COMPUTE_TAU Compute arrival delays for each array element and steering angle.
%
% Eq.(3): tau(theta, r_j) = integral sqrt(1 / c(z)^2 - cos(theta)^2 / c0^2) dz
% Plane-wave approximation: tau(theta, r_j) = (z_j - z0) * sin(theta) / c0
%
% Inputs:
%   theta_vec            1 x Ntheta steering angles [rad]
%   array_depths_m       1 x N array element depths [m]
%   sound_speed_ms       sound-speed profile values [m/s]
%   sound_speed_depth_m  sound-speed profile depths [m]
%   use_plane_wave       true to use plane-wave delays for all angles
%
% Output:
%   tau_matrix           N x Ntheta delay matrix [s]

if nargin < 5
    use_plane_wave = false;
end

num_elements = length(array_depths_m);
num_theta = length(theta_vec);
tau_matrix = zeros(num_elements, num_theta);

[sound_speed_min_ms, sound_channel_idx] = min(sound_speed_ms);
sound_channel_depth_m = sound_speed_depth_m(sound_channel_idx);

tau_plane = ((array_depths_m(:) - sound_channel_depth_m) / sound_speed_min_ms) ...
    * sin(theta_vec);

if use_plane_wave
    tau_matrix = tau_plane;
    return;
end

for theta_idx = 1:num_theta
    theta = theta_vec(theta_idx);
    ray_parameter = cos(theta)^2 / sound_speed_min_ms^2;
    tau_column = zeros(num_elements, 1);
    use_fallback = false;

    for element_idx = 1:num_elements
        element_depth_m = array_depths_m(element_idx);

        if element_depth_m >= sound_channel_depth_m
            depth_mask = sound_speed_depth_m >= sound_channel_depth_m ...
                & sound_speed_depth_m <= element_depth_m;
        else
            depth_mask = sound_speed_depth_m >= element_depth_m ...
                & sound_speed_depth_m <= sound_channel_depth_m;
        end

        depth_seg_m = sound_speed_depth_m(depth_mask);
        sound_speed_seg_ms = sound_speed_ms(depth_mask);

        if length(depth_seg_m) < 2
            tau_column(element_idx) = 0;
            continue;
        end

        integrand_sq = 1 ./ sound_speed_seg_ms.^2 - ray_parameter;
        if any(integrand_sq < 0) || any(~isfinite(integrand_sq))
            use_fallback = true;
            break;
        end

        tau_element = trapz(depth_seg_m, sqrt(integrand_sq));
        if element_depth_m >= sound_channel_depth_m
            tau_column(element_idx) = sign(sin(theta)) * tau_element;
        else
            tau_column(element_idx) = -sign(sin(theta)) * tau_element;
        end
    end

    if use_fallback || any(isnan(tau_column))
        tau_matrix(:, theta_idx) = tau_plane(:, theta_idx);
    else
        tau_matrix(:, theta_idx) = tau_column;
    end
end
end
