function detectedLabels = BSCDDetector(audioPath, varargin)
% BSCDDetector detects events in an audio file using a BSCD-based method.
%
%   detectedLabels = BSCDDetector(audioPath)
%   detectedLabels = BSCDDetector(audioPath, varargin)
%
%   Optional name-value pair arguments:
%       'fcutMin'       : Minimum frequency cutoff (Hz) [default: 40000]
%       'fcutMax'       : Maximum frequency cutoff (Hz) [default: 120000]
%       'fs'            : Sampling frequency (Hz) [default: 250000]
%       'ROIstart'      : ROI start time (s) [default: 50]
%       'ROIlength'     : ROI length (s) [default: 20]
%       'runWholeSignal': Boolean flag; if true, process entire signal [default: true]
%       'plotDetection' : Boolean flag to enable plotting [default: false]
%
%       % BSCD detector parameters:
%       'wlen'      : Window length for BSCD (s) [default: 0.01]
%       'maWindow'  : Moving average window (samples) for smoothing [default: 5000]
%
%       % Noise tracking & Adaptive Threshold parameters:
%       'noiseWindow' : Noise window (samples) for noise floor estimation [default: 256]
%       'localWindow' : Local window (samples) for computing local mean/std [default: 256]
%       'k'           : Scaling factor for adaptive threshold [default: 0.023]
%       'w'           : Weight for local SNR in thresholding [default: 0.994]
%
%       % Post-processing parameters:
%       'minDuration'       : Minimum event duration (s) [default: 0.005]
%       'maxGap'            : Maximum gap (s) to merge events [default: 0.0001]
%       'minEffectivePower' : Minimum effective envelope power [default: 0.0001]
%
%   The function returns detectedLabels, a structure array with fields:
%       .StartTime  : Start time of detected event (s)
%       .EndTime    : End time of detected event (s)
%
%   If outputPath is provided, the detected labels are exported to that file.
%
%   Note: This function assumes that helper functions importLabels,
%         exportLabels, and compareLabels are available in your MATLAB path.
%
%   Example:
%       labels = BSCDDetector('myAudio.wav','myLabels.txt','myDetected.txt',...
%                   'fcutMin',50000, 'plotDetection', true);

%% Input Parsing and Default Settings
p = inputParser;
addRequired(p, 'audioPath', @isstring);

% Basic settings
addParameter(p, 'fcutMin', 40000, @isnumeric);
addParameter(p, 'fcutMax', 120000, @isnumeric);
addParameter(p, 'fs', 250000, @isnumeric);
addParameter(p, 'ROIstart', 50, @isnumeric);
addParameter(p, 'ROIlength', 20, @isnumeric);
addParameter(p, 'runWholeSignal', true, @islogical);

% BSCD detector parameters
addParameter(p, 'wlen', 0.01, @isnumeric);
addParameter(p, 'maWindow', 5000, @isnumeric);

% Noise tracking & Adaptive Threshold parameters
addParameter(p, 'noiseWindow', 256, @isnumeric);
addParameter(p, 'localWindow', 256, @isnumeric);
addParameter(p, 'k', 0.023, @isnumeric);
addParameter(p, 'w', 0.994, @isnumeric);

% Post-Processing parameters
addParameter(p, 'minDuration', 0.005, @isnumeric);
addParameter(p, 'maxGap', 0.0001, @isnumeric);
addParameter(p, 'minEffectivePower', 0.0001, @isnumeric);

parse(p, audioPath, varargin{:});
opts = p.Results;

%% LOAD AUDIO
[audioData, fs_audio] = audioread(opts.audioPath);
if fs_audio ~= opts.fs
    audioData = resample(audioData, opts.fs, fs_audio);
end
t_full = (0:length(audioData)-1) / opts.fs;
audioData = audioData - mean(audioData);
audioData = audioData / max(abs(audioData));

%% SELECT SIGNAL PORTION: ROI or Whole Signal
if opts.runWholeSignal
    xROI = audioData;
    tROI = t_full;
    opts.ROIstart = 0;       % Whole signal starts at 0
    opts.ROIlength = t_full(end);
else
    startIndex = round(opts.ROIstart * opts.fs) + 1;
    endIndex   = round((opts.ROIstart + opts.ROIlength) * opts.fs);
    xROI = audioData(startIndex:endIndex);
    tROI = t_full(startIndex:endIndex);
end

%% BANDPASS FILTER
bpFilter = designfilt('bandpassiir', 'FilterOrder', 12, ...
    'HalfPowerFrequency1', opts.fcutMin, 'HalfPowerFrequency2', opts.fcutMax, ...
    'SampleRate', opts.fs);
filteredAudioSegment = filtfilt(bpFilter, xROI);

%% BSCD CALCULATION
tic;
fprintf('Calculating BSCD...\n');
bscdOut = bscd(filteredAudioSegment.^2, opts.wlen * opts.fs);
toc;
bscdOutMovMean = smoothdata(bscdOut, 'movmean', opts.maWindow);

% Create time axis for BSCD output (using ROI start and ROIlength)
startTime = opts.ROIstart;
endTime = opts.ROIstart + opts.ROIlength;
t_bscd = linspace(startTime, endTime, length(bscdOutMovMean));
powerEnvelope = bscdOutMovMean;

%% DETECTION USING BSCD
optimalThreshold = mean(powerEnvelope) * ones(size(powerEnvelope));
binaryDetections = powerEnvelope > optimalThreshold;
detectionDiff = diff([0, binaryDetections, 0]);
startIndices = find(detectionDiff == 1);
endIndices = find(detectionDiff == -1) - 1;
nEvents = length(startIndices);
detectedLabels = struct('StartTime', cell(nEvents,1), 'EndTime', cell(nEvents,1));
for i = 1:nEvents
    detectedLabels(i).StartTime = t_bscd(startIndices(i));
    detectedLabels(i).EndTime = t_bscd(endIndices(i));
end

%% POST-PROCESSING: Merge events if the gap between them is less than maxGap
if length(detectedLabels) > 1
    mergedEvents = detectedLabels(1);
    for i = 2:length(detectedLabels)
        gap = detectedLabels(i).StartTime - mergedEvents(end).EndTime;
        if gap < opts.maxGap
            mergedEvents(end).EndTime = detectedLabels(i).EndTime;
        else
            mergedEvents = [mergedEvents; detectedLabels(i)]; %#ok<AGROW>
        end
    end
    detectedLabels = mergedEvents;
end
nEvents = length(detectedLabels);

end