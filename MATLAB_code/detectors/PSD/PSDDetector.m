function detectedLabels = PSDDetector(audioPath, labelPath, outputPath, varargin)
% PSDDetector detects events in an audio file using a PSD-based method.
%
%   detectedLabels = PSDDetector(audioPath)
%   detectedLabels = PSDDetector(audioPath,'fcutMin')
%
%   Optional name-value pair arguments can be provided:
%       'fcutMin'       : Minimum frequency cutoff (Hz) [default: 40000]
%       'fcutMax'       : Maximum frequency cutoff (Hz) [default: 120000]
%       'fs'            : Sampling frequency (Hz) [default: 250000]
%       'ROIstart'      : ROI start time (s) [default: 60]
%       'ROIlength'     : ROI length (s) [default: 10]
%       'runWholeSignal': Boolean flag; if true, process entire signal [default: true]
%       'plotDetection' : Boolean flag to enable plotting [default: false]
%
%       % PSD (Welch) parameters:
%       'segmentLength' : Segment length (samples) for Welch PSD [default: 8192]
%       'overlapFactor' : Overlap factor (fraction) [default: 0.59]
%       'maWindow'      : Moving average window (samples) for smoothing [default: 3]
%
%       % Noise tracking & Adaptive Threshold parameters:
%       'noiseWindow'   : Noise window (samples) for noise floor estimation [default: 240]
%       'localWindow'   : Local window (samples) for computing local mean/std [default: 194]
%       'k'             : Scaling factor for adaptive threshold [default: 0.023]
%       'w'             : Weight for local SNR in adaptive thresholding [default: 0.994]
%
%       % Post-processing parameters:
%       'minDuration'       : Minimum event duration (s) [default: 0.0005]
%       'maxGap'            : Maximum gap (s) to merge events [default: 0.001]
%       'minEffectivePower' : Minimum effective envelope power [default: 0.0001]
%
%   The function returns detectedLabels, a structure array with fields:
%       .StartTime  : Start time of detected event (s)
%       .EndTime    : End time of detected event (s)
%
%   If outputPath is provided, the detected labels are also exported to that file.
%
%   Note: This function assumes that helper functions importLabels,
%         exportLabels, and compareLabels are available in your MATLAB path.
%
%   Example:
%       labels = PSDDetector('myAudio.wav', 'myLabels.txt', 'myDetected.txt', 'ROIstart', 30, 'ROIlength', 15);

%% Input Parsing and Default Settings
p = inputParser;
addRequired(p, 'audioPath', @isstring);

% Detector basic settings
addParameter(p, 'fcutMin', 40000, @isnumeric);
addParameter(p, 'fcutMax', 120000, @isnumeric);
addParameter(p, 'fs', 250000, @isnumeric);
addParameter(p, 'ROIstart', 60, @isnumeric);
addParameter(p, 'ROIlength', 10, @isnumeric);
addParameter(p, 'runWholeSignal', true, @islogical);

% PSD (Welch) parameters
addParameter(p, 'segmentLength', 8192, @isnumeric);
addParameter(p, 'overlapFactor', 0.59, @isnumeric);
addParameter(p, 'maWindow', 3, @isnumeric);

% Noise tracking & Adaptive Threshold parameters
addParameter(p, 'noiseWindow', 240, @isnumeric);
addParameter(p, 'localWindow', 194, @isnumeric);
addParameter(p, 'k', 0.023, @isnumeric);
addParameter(p, 'w', 0.994, @isnumeric);

% Post-Processing parameters
addParameter(p, 'minDuration', 0.0005, @isnumeric);
addParameter(p, 'maxGap', 0.001, @isnumeric);
addParameter(p, 'minEffectivePower', 0.0001, @isnumeric);

parse(p, audioPath, varargin{:});
opts = p.Results;

%% LOAD AUDIO
[x, fs_audio] = audioread(opts.audioPath);
if fs_audio ~= opts.fs
    x = resample(x, opts.fs, fs_audio);
end
t_full = (0:length(x)-1) / opts.fs;
x = x - mean(x);
x = x / max(abs(x));

%% SELECT SIGNAL PORTION: ROI or Whole Signal
if opts.runWholeSignal
    xROI = x;
    tROI = t_full;
    opts.ROIstart = 0;        % For whole signal, ROI starts at 0
    opts.ROIlength = t_full(end);
else
    startIndex = round(opts.ROIstart * opts.fs) + 1;
    endIndex   = round((opts.ROIstart + opts.ROIlength) * opts.fs);
    xROI = x(startIndex:endIndex);
    tROI = t_full(startIndex:endIndex);
end

%% PSD ESTIMATION USING WELCH METHOD
nfft = opts.segmentLength;
window = hamming(opts.segmentLength);
[S, F, T_spec] = spectrogram(xROI, window, round(opts.segmentLength * opts.overlapFactor), nfft, opts.fs);
freqMask = (F >= opts.fcutMin) & (F <= opts.fcutMax);
powerEnvelope = sum(abs(S(freqMask, :)).^2, 1);
powerEnvelope = powerEnvelope / max(powerEnvelope);
powerEnvelope = smoothdata(powerEnvelope, 'movmean', opts.maWindow);

%% NOISE TRACKING & EFFECTIVE ENVELOPE
% Estimate noise floor using a moving minimum filter
noiseFloor = movmin(powerEnvelope, opts.noiseWindow);
effectiveEnvelope = powerEnvelope - noiseFloor;
effectiveEnvelope(effectiveEnvelope < 0) = 0;

%% LOCAL SNR CALCULATION
localSNR = effectiveEnvelope ./ (noiseFloor + eps);
localSNR = min(localSNR, 10);

%% ADAPTIVE THRESHOLDING
localMean = movmean(effectiveEnvelope, opts.localWindow);
localStd  = movstd(effectiveEnvelope, opts.localWindow);
optimalThreshold = (localMean + opts.k * localStd) ./ (1 + opts.w * localSNR);

%% DETECTION
binaryDetections = effectiveEnvelope > optimalThreshold;
groupedDetections = diff([0, binaryDetections, 0]);
eventStarts = find(groupedDetections == 1);
eventEnds = find(groupedDetections == -1) - 1;
% Adjust event times by ROIstart
eventTimesStart = T_spec(eventStarts) + opts.ROIstart;
eventTimesEnd   = T_spec(eventEnds) + opts.ROIstart;
nEvents = length(eventTimesStart);
detectedLabels = struct('StartTime', cell(nEvents,1), 'EndTime', cell(nEvents,1));
for i = 1:nEvents
    detectedLabels(i).StartTime = eventTimesStart(i);
    detectedLabels(i).EndTime = eventTimesEnd(i);
end

%% POST-PROCESSING
% Remove events that are too short or have low effective power
validEvents = [];
for i = 1:length(detectedLabels)
    duration = detectedLabels(i).EndTime - detectedLabels(i).StartTime;
    idx = find((T_spec + opts.ROIstart) >= detectedLabels(i).StartTime & (T_spec + opts.ROIstart) <= detectedLabels(i).EndTime);
    avgPower = mean(effectiveEnvelope(idx));
    if duration >= opts.minDuration && avgPower >= opts.minEffectivePower
        validEvents = [validEvents; detectedLabels(i)]; %#ok<AGROW>
    end
end
detectedLabels = validEvents;

% Merge events if the gap between them is less than maxGap
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