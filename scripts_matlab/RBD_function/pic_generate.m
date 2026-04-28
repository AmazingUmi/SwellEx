function [rgb_img, spec_data] = pic_generate(signal, fs, varargin)
%PIC_GENERATE Generate one RGB spectrogram image for a single signal.
%
%   [rgb_img, spec_data] = pic_generate(signal, fs)
%   builds one RGB image from a single time-domain signal segment using:
%     R: mel spectrogram
%     G: CQT spectrogram
%     B: Bark spectrogram
%
%   The defaults in this function are tuned for the current SwellEx
%   project setup, where the signal sampling rate is typically 1500 Hz.
%
%   Optional name-value pairs:
%     'NumBands'   : output image size and spectral band count, default 64
%     'SavePath'   : optional PNG output path
%     'Window'     : spectrogram window for Bark map, default hamming(128)
%     'NOverlap'   : Bark spectrogram overlap, default round(0.75*window)
%     'NFFT'       : Bark spectrogram FFT length, default 256
%     'FMin'       : minimum CQT frequency, default 20 Hz
%     'FMax'       : maximum CQT frequency, default fs/2
%
%   Outputs:
%     rgb_img      : NumBands x NumBands x 3 image in [0, 1]
%     spec_data    : struct with individual mel/cqt/bark images and maps

parser = inputParser;
parser.addRequired('signal', @(x) isnumeric(x) && isvector(x) && ~isempty(x));
parser.addRequired('fs', @(x) isnumeric(x) && isscalar(x) && x > 0);
parser.addParameter('NumBands', 64, @(x) isnumeric(x) && isscalar(x) && x > 0);
parser.addParameter('SavePath', '', @(x) ischar(x) || isstring(x));
parser.addParameter('Window', hamming(128), @(x) isnumeric(x) && isvector(x) && ~isempty(x));
parser.addParameter('NOverlap', [], @(x) isnumeric(x) && isscalar(x) && x >= 0);
parser.addParameter('NFFT', 256, @(x) isnumeric(x) && isscalar(x) && x > 0);
parser.addParameter('FMin', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x > 0));
parser.addParameter('FMax', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x > 0));
parser.parse(signal, fs, varargin{:});

num_bands = parser.Results.NumBands;
save_path = char(parser.Results.SavePath);
window = parser.Results.Window(:);
nfft = parser.Results.NFFT;

if isempty(parser.Results.NOverlap)
    noverlap = round(0.75 * length(window));
else
    noverlap = parser.Results.NOverlap;
end

if isempty(parser.Results.FMin)
    fmin = 20;
else
    fmin = parser.Results.FMin;
end

if isempty(parser.Results.FMax)
    fmax = fs / 2;
else
    fmax = parser.Results.FMax;
end

assert(fmin < fmax, 'FMin must be smaller than FMax.');
assert(noverlap < length(window), 'NOverlap must be smaller than the window length.');

s = signal(:);

% 1) Mel spectrogram
[mel_spec, mel_freq, mel_time] = melSpectrogram(s, fs, 'NumBands', num_bands);
mel_spec_db = 10 * log10(mel_spec + eps);
mel_img = imresize(mat2gray(flipud(mel_spec_db)), [num_bands, num_bands]);

% 2) CQT spectrogram
num_octaves = log2(fmax / fmin);
bins_per_octave = max(12, round(num_bands / num_octaves));
cqt_obj = cqt(s, 'SamplingFrequency', fs, ...
    'BinsPerOctave', bins_per_octave, ...
    'FrequencyLimits', [fmin, fmax]);
cqt_spec = abs(cqt_obj.c);
cqt_spec = cqt_spec(1:ceil(size(cqt_spec, 1) / 2), :);
cqt_spec_db = 10 * log10(cqt_spec + eps);
cqt_img = imresize(mat2gray(flipud(cqt_spec_db)), [num_bands, num_bands]);

% 3) Bark spectrogram
[S, F, T] = spectrogram(s, window, noverlap, nfft, fs);
bark_f = 13 * atan(0.00076 * F) + 3.5 * atan((F / 7500).^2);
edges = linspace(min(bark_f), max(bark_f), num_bands + 1);
bark_spec = zeros(num_bands, length(T));
for m = 1:num_bands
    idx = bark_f >= edges(m) & bark_f < edges(m + 1);
    if any(idx)
        bark_spec(m, :) = sum(abs(S(idx, :)), 1);
    end
end
bark_spec_db = 10 * log10(bark_spec + eps);
bark_img = imresize(mat2gray(flipud(bark_spec_db)), [num_bands, num_bands]);

rgb_img = cat(3, mel_img, cqt_img, bark_img);

spec_data = struct( ...
    'mel_img', mel_img, ...
    'cqt_img', cqt_img, ...
    'bark_img', bark_img, ...
    'mel_spec_db', mel_spec_db, ...
    'cqt_spec_db', cqt_spec_db, ...
    'bark_spec_db', bark_spec_db, ...
    'mel_freq', mel_freq, ...
    'mel_time', mel_time, ...
    'bark_freq', F, ...
    'bark_time', T, ...
    'window', window, ...
    'noverlap', noverlap, ...
    'nfft', nfft, ...
    'fmin', fmin, ...
    'fmax', fmax, ...
    'num_bands', num_bands);

if ~isempty(save_path)
    imwrite(rgb_img, save_path, 'Compression', 'none');
end
end
