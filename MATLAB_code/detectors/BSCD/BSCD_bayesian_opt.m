clc; close all; clear;

%% =============== SELECT AUDIO FILE ===============
id = 1;
switch id
    case 1
        audioFile = "LPS-SI2homo-mH02-I04-USV";
    case 2
        audioFile = "LPS-SI2homo-mK01-L03-USV";
    case 3
        audioFile = "LPS-SI2homo-mL01-M02-USV";
    case 4
        audioFile = "LPS-SI2homo-mO02-S03-USV";
    case 5
        audioFile = "LPS-SI2homo-mR03-T01-USV";
    case 6
        audioFile = "LPS-SI2homo-mR04-S01-USV"; 
end
disp("------------------------");
disp(audioFile);

%% =============== PARAMETERS (Best-Fit / Initial) ===============
fcutMin       = 40000;    % Minimum frequency cutoff (Hz)
fcutMax       = 120000;   % Maximum frequency cutoff (Hz)
fs            = 250000;   % Sampling frequency (Hz)
ROIstart      = 100;       % ROI start time (s)
ROIlength     = 10;       % ROI length (s)
runWholeSignal = false;    % Process only the ROI

% Flag to control plotting (set false to run optimization only)
plotDetection = false;  

% Initial BSCD detector parameters (will be optimized):
wlen_init     = 0.01;     % BSCD window length in seconds
maW_init      = 5000;     % Moving average window for smoothing (samples)
noiseW_init   = 256;      % Noise window for noise floor estimation (samples)
localW_init   = 256;      % Local window for computing mean and std (samples)
k_init        = 0.023;    % Scaling factor for adaptive threshold
w_init        = 0.994;    % Weight for local SNR in thresholding

% Post-Processing Parameters (fixed)
minDuration = 0.005;      % Minimum event duration (s)
maxGap      = 0.0001;     % Maximum gap between events to merge (s)
minEffectivePower = 0.0001;  % Minimum average effective envelope power for an event

%% ==================== SETTINGS ====================
enablePlotting = true;
testStartTime = datetime('now','TimeZone','local','Format','yyyy-MM-dd HH:mm:ss');
fprintf('Test start time: %s\n', testStartTime);

% File Paths
basePath  = "/Users/gazda/Documents/CTU/Masters/Masters thesis/";
datapath  = fullfile(basePath, "data");
audioPath = fullfile(datapath, "usv_audio/denoise", audioFile + "_denoise.wav");
labelPath = fullfile(datapath, "labels", audioFile + "-IVojt.txt");
outputPath = fullfile(datapath, audioFile + "_detected.txt");
fprintf('Filename: %s\n', audioFile);

%% ==================== LOAD DATA AND EXTRACT ROI ====================
[audioData, fs_audio] = audioread(audioPath);
if fs_audio ~= fs
    audioData = resample(audioData, fs, fs_audio);
end
t_full = (0:length(audioData)-1)/fs;
% Define ROI indices
if runWholeSignal
    audioSegment = audioData;
    tROI = t_full;
else
    startIndex = round(ROIstart * fs) + 1;
    stopIndex  = round((ROIstart + ROIlength) * fs);
    audioSegment = audioData(startIndex:stopIndex, :);
    tROI = t_full(startIndex:stopIndex);
end
xROI = audioSegment;

% Import provided (ground truth) labels and save ROI-filtered version
providedLabelsFull = importLabels(labelPath, fs);
if runWholeSignal
    providedLabelsROI = providedLabelsFull;
else
    ROIend = ROIstart + ROIlength;
    providedLabelsROI = providedLabelsFull(arrayfun(@(x) (x.StartTime >= ROIstart && x.EndTime <= ROIend), providedLabelsFull));
end
tempProvidedFile = fullfile(tempdir, "provided_labels.txt");
saveDetectedLabels(providedLabelsROI, tempProvidedFile);

%% ==================== BANDPASS FILTER ====================
bpFilter = designfilt('bandpassiir', 'FilterOrder', 12, ...
    'HalfPowerFrequency1', fcutMin, 'HalfPowerFrequency2', fcutMax, ...
    'SampleRate', fs);
filteredAudioSegment = filtfilt(bpFilter, audioSegment);

%% ==================== DEFINE OPTIMIZATION VARIABLES ====================
% We optimize the following BSCD detector parameters:
%   - wlen: BSCD window length in seconds.
%   - maW: Moving average window (integer samples).
%   - noiseW: Noise window for moving median (integer samples).
%   - localW: Window for local mean/std (integer samples).
%   - k: Scaling factor for adaptive threshold.
%   - w: Weight for local SNR in adaptive threshold.
vars = [...
    optimizableVariable('wlen', [0.005, 0.05]),...
    optimizableVariable('maW', [1000, 10000], 'Type', 'integer'),...
    optimizableVariable('noiseW', [100, 500], 'Type', 'integer'),...
    optimizableVariable('localW', [25, 200], 'Type', 'integer'),...
    optimizableVariable('k', [0.01, 0.2]),...
    optimizableVariable('w', [0, 1])];

%% ==================== RUN BAYESIAN OPTIMIZATION ====================
% The objective function runs the BSCD detector with candidate parameters,
% compares the detected events with the provided labels, and returns negative F1-score.
objFun = @(optVars) detectorObjective_BSCD(optVars, filteredAudioSegment, fs, ROIstart, tempProvidedFile);
results = bayesopt(objFun, vars, 'MaxObjectiveEvaluations', 30, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', 'Verbose', 1);

bestParams = bestPoint(results);
bestF1 = -results.MinObjective;  % because the objective returns -F1

fprintf('\nOptimal Parameters Found:\n');
fprintf('wlen: %.4f s\n', bestParams.wlen);
fprintf('maW: %d samples\n', bestParams.maW);
fprintf('Noise Window: %d samples\n', bestParams.noiseW);
fprintf('Local Window: %d samples\n', bestParams.localW);
fprintf('k Factor: %.3f\n', bestParams.k);
fprintf('w Factor: %.3f\n', bestParams.w);
fprintf('Best F1-Score: %.3f\n', bestF1);

%% ==================== FINAL DETECTION USING BEST PARAMETERS ====================
% Run the BSCD detector using the optimized parameters
bscdOut = bscd(filteredAudioSegment.^2, bestParams.wlen*fs);
bscdOutMovMean = smoothdata(bscdOut, 'movmean', bestParams.maW);
t_bscd = linspace(ROIstart, ROIstart + ROIlength, length(bscdOutMovMean));
powerEnvelope = bscdOutMovMean;

% Adaptive noise tracking
noiseFloor = movmedian(powerEnvelope, bestParams.noiseW);
effectiveEnvelope = powerEnvelope - noiseFloor;
effectiveEnvelope(effectiveEnvelope < 0) = 0;

% Compute local SNR (with clipping)
localSNR = effectiveEnvelope ./ (noiseFloor + eps);
localSNR = min(localSNR, 10);

localMean = movmean(effectiveEnvelope, bestParams.localW);
localStd  = movstd(effectiveEnvelope, bestParams.localW);
adaptiveThreshold = (localMean + bestParams.k * localStd) ./ (1 + bestParams.w * localSNR);

binaryDetections = effectiveEnvelope > adaptiveThreshold;
groupedDetections = diff([0, binaryDetections, 0]);
eventStarts = find(groupedDetections == 1);
eventEnds = find(groupedDetections == -1) - 1;
% Create detected labels (adjusting time to absolute scale)
eventTimesStart = t_bscd(eventStarts);
eventTimesEnd = t_bscd(eventEnds);
nEvents = length(eventTimesStart);
detectedLabels = struct('StartTime', cell(nEvents,1), 'EndTime', cell(nEvents,1));
for i = 1:nEvents
    detectedLabels(i).StartTime = eventTimesStart(i);
    detectedLabels(i).EndTime = eventTimesEnd(i);
end

tempDetectedFile = fullfile(tempdir, "detected_labels.txt");
saveDetectedLabels(detectedLabels, tempDetectedFile);
finalStats = compareLabels(tempProvidedFile, tempDetectedFile, fs);

fprintf('\nFinal Detection Performance with Optimal Parameters:\n');
fprintf('True Positives: %d\n', finalStats.TruePositives);
fprintf('False Positives: %d\n', finalStats.FalsePositives);
fprintf('False Negatives: %d\n', finalStats.FalseNegatives);
fprintf('Precision: %.3f\n', finalStats.Precision);
fprintf('Recall: %.3f\n', finalStats.Recall);
fprintf('F1-Score: %.3f\n', finalStats.F1Score);

%% ==================== OUTPUT: Save Final Detection Results ====================
fprintf("\nDetection complete. Total events detected: %d\n", nEvents);
fileID = fopen(outputPath, 'w');
fprintf(fileID, "Detection complete. Total events detected: %d\n", nEvents);
for i = 1:nEvents
    fprintf(fileID, "Event %d: Start Time = %.3f s, End Time = %.3f s\n", i, detectedLabels(i).StartTime, detectedLabels(i).EndTime);
end
fclose(fileID);
fprintf("Detection results saved to: %s\n", outputPath);

%% ==================== Helper Function: saveDetectedLabels ====================
function saveDetectedLabels(labels, filePath)
    fid = fopen(filePath, 'w');
    for idx = 1:length(labels)
        fprintf(fid, '%.6f\t%.6f\tdetected\n', labels(idx).StartTime, labels(idx).EndTime);
        fprintf(fid, '0\t0\t\n');
    end
    fclose(fid);
end

%% ==================== Nested Objective Function for Bayesian Optimization ====================
function objective = detectorObjective_BSCD(optVars, filteredAudioSegment, fs, ROIstart, tempProvidedFile)
    % Use candidate parameters for BSCD detector:
    %   optVars.wlen, optVars.maW, optVars.noiseW, optVars.localW, optVars.k, optVars.w
    
    % Compute BSCD on the filtered signal (squared)
    bscdOut = bscd(filteredAudioSegment.^2, optVars.wlen * fs);
    % Smooth BSCD output with moving average window
    bscdOutMovMean = smoothdata(bscdOut, 'movmean', optVars.maW);
    
    % Create time axis for BSCD output (using ROI start & length)
    numSamples = length(bscdOutMovMean);
    t_bscd = linspace(ROIstart, ROIstart + (numSamples/fs), numSamples);
    powerEnvelope = bscdOutMovMean;
    
    % Noise tracking: Estimate noise floor using moving median
    noiseFloor = movmedian(powerEnvelope, optVars.noiseW);
    effectiveEnvelope = powerEnvelope - noiseFloor;
    effectiveEnvelope(effectiveEnvelope < 0) = 0;
    
    % Compute local SNR (clip extreme values)
    localSNR = effectiveEnvelope ./ (noiseFloor + eps);
    localSNR = min(localSNR, 10);
    
    % Adaptive thresholding: combine local statistics with SNR
    localMean = movmean(effectiveEnvelope, optVars.localW);
    localStd  = movstd(effectiveEnvelope, optVars.localW);
    adaptiveThreshold = (localMean + optVars.k * localStd) ./ (1 + optVars.w * localSNR);
    
    % Binary detection
    binaryDetections = effectiveEnvelope > adaptiveThreshold;
    groupedDetections = diff([0, binaryDetections, 0]);
    eventStarts = find(groupedDetections == 1);
    eventEnds = find(groupedDetections == -1) - 1;
    eventTimesStart = t_bscd(eventStarts);
    eventTimesEnd = t_bscd(eventEnds);
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
    
    % Objective is negative F1 (since we maximize F1)
    objective = -F1;
end