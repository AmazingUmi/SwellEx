% extract_sig_from_sio.m
% Read a multi-channel SIO file and export each VLA channel to a MAT file.

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

cd(script_dir);
addpath(script_dir);
addpath(genpath(function_dir));
clear tmp;

%% Input and output paths
sio_file = fullfile(project_dir, 'events', 'S5', 'J1312315.vla.21els.sio');
output_dir = fullfile(project_dir, 'events', 'S5', 'vla_matfiles');
output_name_fmt = 'S5_VLA_NO_%d.mat';

if ~isfolder(output_dir)
    mkdir(output_dir);
end

%% Read options for sioread.m
p1 = 1;
npi = 0;
channel_idx_vec = 1:21;

%% Export channels
fprintf('Reading SIO: %s\n', sio_file);

for channel_list_idx = 1:numel(channel_idx_vec)
    channel_idx = channel_idx_vec(channel_list_idx);
    fprintf('p1=%d, npi=%d, channel=[%s]\n', p1, npi, num2str(channel_idx));

    x = sioread(sio_file, p1, npi, channel_idx);
    fprintf('Read done. size(x) = [%d, %d] (points x channels)\n', size(x, 1), size(x, 2));

    output_file = fullfile(output_dir, sprintf(output_name_fmt, channel_idx));
    save(output_file, 'x');
end

fprintf('All done.\n');
