clc; clear; close all;

%% FIXED PARAMETERS (for candidate search)
fcutMin       = 40000;   % Minimum frequency cutoff (Hz)
fcutMax       = 120000;  % Maximum frequency cutoff (Hz)
fs            = 250000;  % Sampling frequency (Hz)
ROIstart      = 100;     % ROI start time (s)
ROIlength     = 60;      % ROI length (s)
runWholeSignal = true;  % Process only the ROI

% cely signal
% 
% Optimal Parameters Found:
% Segment Length: 14502 samples
% Overlap Factor: 0.57
% Moving Average Window: 3 samples
% Noise Window: 242 samples
% Adaptive Local Window: 192 samples
% k Factor: 0.020
% w Factor: 0.995
% Best F1-Score: 0.621
% Statistics based on Midpoint Criterion:
%     TotalProvidedLabels: 1451
%     TotalDetectedLabels: 1479
%           TruePositives: 910
%          FalsePositives: 569
%          FalseNegatives: 541
%               Precision: 0.6153
%                  Recall: 0.6272
%                 F1Score: 0.6212


%% Load Audio and Extract ROI
basePath  = "/Users/gazda/Documents/CTU/Masters/Masters thesis/";
datapath  = fullfile(basePath, "data");
audioFile = "LPS-SI2homo-mH02-I04-USV";
audioPath = fullfile(datapath, "usv_audio", audioFile + ".wav");

[x, fs_audio] = audioread(audioPath);
if fs_audio ~= fs
    x = resample(x, fs, fs_audio);
end
t = (0:length(x)-1) / fs;
x = x - mean(x);
x = x / max(abs(x));

if runWholeSignal
    xROI = x;
    tROI = t;
    ROIstart = 0;
    ROIlength = t(end);
else
    startIndex = round(ROIstart * fs) + 1;
    endIndex   = round((ROIstart + ROIlength) * fs);
    xROI = x(startIndex:endIndex);
    tROI = t(startIndex:endIndex);
end

%% Load Provided (Ground Truth) Labels and Save ROI-Filtered Version
audioFile = "LPS-SI2homo-mH02-I04-USV";
labelPath = fullfile(datapath, "labels", audioFile + "-IVojt.txt");
providedLabelsFull = importLabels(labelPath, fs);
if runWholeSignal
    providedLabelsROI = providedLabelsFull;
else
    ROIend = ROIstart + ROIlength;
    providedLabelsROI = providedLabelsFull(arrayfun(@(x) (x.StartTime >= ROIstart && x.EndTime <= ROIend), providedLabelsFull));
end
tempProvidedFile = fullfile(tempdir, "provided_labels_ROI.txt");
saveDetectedLabels(providedLabelsROI, tempProvidedFile);

%% Define Optimization Variables for Bayesian Optimization
% We optimize:
% - segLen: PSD segment length (integer)
% - overlap: Overlap factor (continuous)
% - maW: Moving average window for smoothing (integer)
% - noiseW: Noise window for noise floor estimation (integer)
% - localW: Window for adaptive threshold local statistics (integer)
% - k: Scaling factor for adaptive threshold (continuous)
% - w: Weight for local SNR in the adaptive threshold (continuous)
vars = [...
    optimizableVariable('segLen', [8192, 16384], 'Type', 'integer'),...
    optimizableVariable('overlap', [0.25, 0.75]),...
    optimizableVariable('maW', [1, 5], 'Type', 'integer'),...
    optimizableVariable('noiseW', [150, 250], 'Type', 'integer'),...
    optimizableVariable('localW', [25, 200], 'Type', 'integer'),...
    optimizableVariable('k', [0.01, 0.2]),...
    optimizableVariable('w', [0, 1])];

%% Run Bayesian Optimization
% Pass required extra variables to the objective function.
objFun = @(optVars) detectorObjective(optVars, xROI, fs, ROIstart, fcutMin, fcutMax, tempProvidedFile);
results = bayesopt(objFun, vars, ...
    'MaxObjectiveEvaluations', 30, 'AcquisitionFunctionName', 'expected-improvement-plus', 'Verbose', 1);

bestParams = bestPoint(results);
bestF1 = -results.MinObjective;  % Because the objective returns -F1

fprintf('\nOptimal Parameters Found:\n');
fprintf('Segment Length: %d samples\n', bestParams.segLen);
fprintf('Overlap Factor: %.2f\n', bestParams.overlap);
fprintf('Moving Average Window: %d samples\n', bestParams.maW);
fprintf('Noise Window: %d samples\n', bestParams.noiseW);
fprintf('Adaptive Local Window: %d samples\n', bestParams.localW);
fprintf('k Factor: %.3f\n', bestParams.k);
fprintf('w Factor: %.3f\n', bestParams.w);
fprintf('Best F1-Score: %.3f\n', bestF1);

%% Final Detection Using Best Parameters (on ROI)
currentWindow = hamming(bestParams.segLen);
[S, F, T_spec] = spectrogram(xROI, currentWindow, round(bestParams.segLen * bestParams.overlap), bestParams.segLen, fs);
freqMask = (F >= fcutMin) & (F <= fcutMax);
powerEnvelope = sum(abs(S(freqMask, :)).^2, 1);
powerEnvelope = powerEnvelope / max(powerEnvelope);
powerEnvelope = smoothdata(powerEnvelope, 'movmean', bestParams.maW);

% Adaptive Noise Tracking using Median Filtering for Noise Floor Estimation
noiseFloor = movmedian(powerEnvelope, bestParams.noiseW);
effectiveEnvelope = powerEnvelope - noiseFloor;
effectiveEnvelope(effectiveEnvelope < 0) = 0;

% Local SNR (clip to avoid extreme values)
localSNR = effectiveEnvelope ./ (noiseFloor + eps);
localSNR = min(localSNR, 10);

% Adaptive thresholding: use the new formula incorporating local SNR:
% adaptiveThreshold = (localMean + k*localStd) ./ (1 + w*localSNR)
localMean = movmean(effectiveEnvelope, bestParams.localW);
localStd  = movstd(effectiveEnvelope, bestParams.localW);
adaptiveThreshold = (localMean + bestParams.k * localStd) ./ (1 + bestParams.w * localSNR);

binaryDetections = effectiveEnvelope > adaptiveThreshold;
groupedDetections = diff([0, binaryDetections, 0]);
eventStarts = find(groupedDetections == 1);
eventEnds = find(groupedDetections == -1) - 1;
eventTimesStart = T_spec(eventStarts) + ROIstart;
eventTimesEnd = T_spec(eventEnds) + ROIstart;
nEvents = length(eventTimesStart);
detectedLabels = struct('StartTime', cell(nEvents,1), 'EndTime', cell(nEvents,1));
for i = 1:nEvents
    detectedLabels(i).StartTime = eventTimesStart(i);
    detectedLabels(i).EndTime = eventTimesEnd(i);
end

tempDetectedFile = fullfile(tempdir, "detected_labels.txt");
saveDetectedLabels(detectedLabels, tempDetectedFile);
finalStats = compareLabels(tempProvidedFile, tempDetectedFile, fs);

fprintf('\nFinal Detection Performance with Optimal Parameters & Enhanced Noise Tracking:\n');
fprintf('True Positives: %d\n', finalStats.TruePositives);
fprintf('False Positives: %d\n', finalStats.FalsePositives);
fprintf('False Negatives: %d\n', finalStats.FalseNegatives);
fprintf('Precision: %.3f\n', finalStats.Precision);
fprintf('Recall: %.3f\n', finalStats.Recall);
fprintf('F1-Score: %.3f\n', finalStats.F1Score);

%% OUTPUT: Save Final Detection Results to File
outputPath = fullfile(datapath, audioFile + "_detected.txt");
fprintf("\nDetection complete. Total events detected: %d\n", nEvents);
for i = 1:nEvents
    fprintf("Event %d: Start Time = %.3f s, End Time = %.3f s\n", i, detectedLabels(i).StartTime, detectedLabels(i).EndTime);
end

fileID = fopen(outputPath, 'w');
fprintf(fileID, "Detection complete. Total events detected: %d\n", nEvents);
for i = 1:nEvents
    fprintf(fileID, "Event %d: Start Time = %.3f s, End Time = %.3f s\n", i, detectedLabels(i).StartTime, detectedLabels(i).EndTime);
end
fclose(fileID);
fprintf("Detection results saved to: %s\n", outputPath);

%% Helper Function: saveDetectedLabels
function saveDetectedLabels(labels, filePath)
    fid = fopen(filePath, 'w');
    for idx = 1:length(labels)
        fprintf(fid, '%.6f\t%.6f\tdetected\n', labels(idx).StartTime, labels(idx).EndTime);
        fprintf(fid, '0\t0\t\n');
    end
    fclose(fid);
end

%% Nested Objective Function for Bayesian Optimization
function objective = detectorObjective(optVars, xROI, fs, ROIstart, fcutMin, fcutMax, tempProvidedFile)
    % Unpack variables
    segLen = optVars.segLen;
    overlap = optVars.overlap;
    maW = optVars.maW;
    noiseW = optVars.noiseW;
    localW = optVars.localW;
    kVal = optVars.k;
    wVal = optVars.w;
    
    % Compute spectrogram using current PSD parameters
    currentWindow = hamming(segLen);
    [S, F, T_spec] = spectrogram(xROI, currentWindow, round(segLen * overlap), segLen, fs);
    freqMask = (F >= fcutMin) & (F <= fcutMax);
    powerEnvelope = sum(abs(S(freqMask, :)).^2, 1);
    powerEnvelope = powerEnvelope / max(powerEnvelope);
    powerEnvelope = smoothdata(powerEnvelope, 'movmean', maW);
    
    % Noise tracking: Use moving median for adaptive noise estimation
    noiseFloor = movmedian(powerEnvelope, noiseW);
    effectiveEnvelope = powerEnvelope - noiseFloor;
    effectiveEnvelope(effectiveEnvelope < 0) = 0;
    
    % Compute local SNR (clipping extreme values)
    localSNR = effectiveEnvelope ./ (noiseFloor + eps);
    localSNR = min(localSNR, 10);
    
    % Adaptive thresholding incorporating local SNR:
    localMean = movmean(effectiveEnvelope, localW);
    localStd  = movstd(effectiveEnvelope, localW);
    adaptiveThreshold = (localMean + kVal * localStd) ./ (1 + wVal * localSNR);
    
    % Binary detection
    binaryDetections = effectiveEnvelope > adaptiveThreshold;
    groupedDetections = diff([0, binaryDetections, 0]);
    eventStarts = find(groupedDetections == 1);
    eventEnds = find(groupedDetections == -1) - 1;
    eventTimesStart = T_spec(eventStarts) + ROIstart;
    eventTimesEnd = T_spec(eventEnds) + ROIstart;
    nEvents = length(eventTimesStart);
    detectedLabels = struct('StartTime', cell(nEvents,1), 'EndTime', cell(nEvents,1));
    for j = 1:nEvents
        detectedLabels(j).StartTime = eventTimesStart(j);
        detectedLabels(j).EndTime = eventTimesEnd(j);
    end
    
    tempDetectedFile = fullfile(tempdir, "detected_labels.txt");
    saveDetectedLabels(detectedLabels, tempDetectedFile);
    
    % Evaluate detection performance using compareLabels
    stats = compareLabels(tempProvidedFile, tempDetectedFile, fs);
    F1 = stats.F1Score;
    
    % Return negative F1 as objective (for minimization)
    objective = -F1;
end