%% Environment setup
clear; close all; clc;

try
    tmp = matlab.desktop.editor.getActive;
    script_dir = fileparts(tmp.Filename);
catch
    script_dir = fileparts(mfilename('fullpath'));
end

project_dir = fileparts(fileparts(script_dir));

cd(script_dir);
addpath(script_dir);
clear tmp;

%% Output path
output_file = fullfile(project_dir, 'source', 'SourceInfo.mat');
output_dir = fileparts(output_file);
if ~isfolder(output_dir)
    mkdir(output_dir);
end

%% Source metadata
DeepSource = struct();
DeepSource.name = 'T-49-13';
DeepSource.depth = 54;
DeepSource.frequencies = [49 64 79 94 112 130 148 166 201 235 283 338 388;
                          52 67 82 97 115 133 151 169 204 238 286 341 391;
                          55 70 85 100 118 136 154 172 207 241 289 344 394;
                          58 73 88 103 121 139 157 175 210 244 292 347 397;
                          61 76 91 106 124 142 160 178 213 247 295 350 400];
DeepSource.levels = [158 132 128 124 120];
DeepSource.notes = ['Deep source was towed at a depth of about 54 m. ', ...
    'It transmitted numerous tonals of various source levels between ', ...
    '49 Hz and 400 Hz. This tonal set is known as T-49-13. The ', ...
    'T-49-13 tonal pattern consists of 5 sets of 13 tones. Each set ', ...
    'of 13 tones spans the frequencies between 49Hz and 400Hz. The ', ...
    'first set of 13 tones is projected at maximum level and is ', ...
    'referred to as the "High Tonal Set." These tones are projected ', ...
    'with transmitted levels of approximately 158 dB. The second set ', ...
    'of tones are projected with levels of approximately 132 dB. The ', ...
    'subsequent sets (3rd, 4th, and 5th) are each projected 4 dB down ', ...
    'from the previous set.'];

ShallowSource = struct();
ShallowSource.name = 'C-109-9S';
ShallowSource.depth = 9;
ShallowSource.frequencies = [109 127 145 163 198 232 280 335 385];
ShallowSource.level = 158;
ShallowSource.notes = ['Shallow source was towed at a depth of about 9m. ', ...
    'It transmitted 9 frequencies between 109 Hz and 385 Hz, known as ', ...
    'the C-109-9S tonal set.'];

Noise = struct();
Noise.name = 'NoiseFrequencies';
Noise.frequencies = [62 77 92 107 125 143 161 179 214 248 296 351 401];
Noise.level = [];
Noise.notes = ['Some initial post-processing utilized so-called "noise ', ...
    'frequencies." Those frequencies chosen to be representative of the ', ...
    'noise field are listed below:'];

save(output_file, 'DeepSource', 'ShallowSource', 'Noise');
