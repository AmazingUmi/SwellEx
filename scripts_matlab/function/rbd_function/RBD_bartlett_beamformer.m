function beam_output = RBD_bartlett_beamformer(signal_freq_seg, omega, tau_matrix, num_elements)
%RBD_BARTLETT_BEAMFORMER Compute Bartlett beamformer output.
%
% Eq.(2): B(omega, theta) = (1 / N) * sum_j exp(-i * omega * tau_j) * P_j(omega)
%
% Inputs:
%   signal_freq_seg N x Nf frequency-domain array data
%   omega           1 x Nf angular-frequency vector [rad/s]
%   tau_matrix      N x Ntheta steering delay matrix [s]
%   num_elements    total number of array elements
%
% Output:
%   beam_output     Nf x Ntheta Bartlett beamformer output

num_freq_bins = size(signal_freq_seg, 2);
num_theta = size(tau_matrix, 2);
beam_output = zeros(num_freq_bins, num_theta);

for theta_idx = 1:num_theta
    tau_column = tau_matrix(:, theta_idx);
    valid_idx = ~isnan(tau_column);
    num_valid = sum(valid_idx);

    if num_valid == 0
        beam_output(:, theta_idx) = NaN;
        continue;
    end

    if num_valid < num_elements
        warning(['Steering angle %d uses only %d/%d valid elements ', ...
            'because some delays are invalid.'], ...
            theta_idx, num_valid, num_elements);
    end

    tau_valid = tau_column(valid_idx);
    signal_freq_valid = signal_freq_seg(valid_idx, :);
    steering_matrix = exp(-1j * tau_valid * omega);

    beam_output(:, theta_idx) = (1 / num_valid) * ...
        sum(steering_matrix .* signal_freq_valid, 1).';
end
end
