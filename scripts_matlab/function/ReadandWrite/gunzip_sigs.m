% gunzip_all_here.m
% Unzip all `.gz` files in the current directory.

clear; clc;

gz_list = dir('*.gz');
if isempty(gz_list)
    fprintf('No .gz files found in %s\n', pwd);
    return;
end

fprintf('Working directory: %s\n', pwd);
fprintf('Found %d .gz files. Starting extraction...\n', numel(gz_list));

num_ok = 0;
num_fail = 0;

for file_idx = 1:numel(gz_list)
    gz_name = gz_list(file_idx).name;
    try
        gunzip(gz_name, pwd);
        num_ok = num_ok + 1;
        fprintf('[%d/%d] OK   %s\n', file_idx, numel(gz_list), gz_name);
    catch ME
        num_fail = num_fail + 1;
        fprintf('[%d/%d] FAIL %s\n  %s\n', ...
            file_idx, numel(gz_list), gz_name, ME.message);
    end
end

fprintf('Finished. OK: %d, FAIL: %d\n', num_ok, num_fail);
