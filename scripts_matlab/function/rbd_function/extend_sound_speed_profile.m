function [depth_ext_m, sound_speed_ext_ms] = extend_sound_speed_profile( ...
    depth_m, sound_speed_ms, depth_max_required_m)
%EXTEND_SOUND_SPEED_PROFILE Extend a sound-speed profile to the required depth.
%
% Inputs:
%   depth_m               sound-speed profile depths [m]
%   sound_speed_ms        sound-speed profile values [m/s]
%   depth_max_required_m  maximum required depth [m]
%
% Outputs:
%   depth_ext_m           extended depth vector [m]
%   sound_speed_ext_ms    extended sound-speed vector [m/s]

depth_max_profile_m = max(depth_m);

if depth_max_profile_m >= depth_max_required_m
    depth_ext_m = depth_m;
    sound_speed_ext_ms = sound_speed_ms;
    return;
end

warning(['Sound-speed profile reaches %.1f m, but the array requires ', ...
    '%.1f m. Extending the profile with a constant bottom sound speed.'], ...
    depth_max_profile_m, depth_max_required_m);

sound_speed_bottom_ms = sound_speed_ms(end);
depth_pad_m = (depth_max_profile_m + 0.5:0.5:depth_max_required_m).';
sound_speed_pad_ms = repmat(sound_speed_bottom_ms, length(depth_pad_m), 1);

depth_ext_m = [depth_m(:); depth_pad_m];
sound_speed_ext_ms = [sound_speed_ms(:); sound_speed_pad_ms];

fprintf(['Extended sound-speed profile from %.1f m to %.1f m. ', ...
    'Bottom sound speed = %.2f m/s.\n'], ...
    depth_max_profile_m, depth_max_required_m, sound_speed_bottom_ms);
end
