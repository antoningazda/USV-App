clc; close all; clear;

%% FIXED PARAMETERS (for candidate search)
fcutMin       = 40000;   % Minimum frequency cutoff (Hz)
fcutMax       = 120000;  % Maximum frequency cutoff (Hz)
fs            = 250000;  % Sampling frequency (Hz)
ROIstart      = 100;     % ROI start time (s)
ROIlength     = 20;      % ROI length (s)
runWholeSignal = true;  % Process only the ROI

% Noise tracking parameter (fixed)
noiseWindow   = 50;      % Window (in samples) for estimating noise floor

%% Candidate PSD Parameters
candidateSegmentLengths = [8192, 16384]; % Candidate segment lengths (FFT lengths)
candidateOverlapFactors = [0.25, 0.4, 0.5, 0.6, 0.75];       % Candidate overlap fractions
candidateMaWindows      = [1, 2, 3, 4, 5];                    % Moving average window sizes (samples)

%% Candidate Adaptive Threshold Parameters
% The adaptive threshold will be computed on the effective envelope:
% effectiveEnvelope = powerEnvelope - noiseFloor
% adaptiveThreshold = localMean(effectiveEnvelope) + k * localStd(effectiveEnvelope)
candidateLocalWindows = [25, 50, 75, 100];  % Candidate local window sizes (samples)
candidateK            = [0.3, 0.4, 0.5, 0.6, 0.7];  % Candidate scaling factors

%% File Paths
basePath  = "/Users/gazda/Documents/CTU/Masters/Masters thesis/";
datapath  = fullfile(basePath, "data");
audioFile = "LPS-SI2homo-mH02-I04-USV";
audioPath = fullfile(datapath, "usv_audio", audioFile + ".wav");
labelPath = fullfile(datapath, "labels", audioFile + "-IVojt.txt");
outputPath = fullfile(datapath, audioFile + "_detected.txt");

%% LOAD AUDIO
[x, fs_audio] = audioread(audioPath);
if fs_audio ~= fs
    x = resample(x, fs, fs_audio);
end
t = (0:length(x)-1) / fs;
x = x - mean(x);
x = x / max(abs(x));

%% SELECT SIGNAL PORTION: ROI (since runWholeSignal = false)
startIndex = round(ROIstart * fs) + 1;
endIndex   = round((ROIstart + ROIlength) * fs);
xROI = x(startIndex:endIndex);
tROI = t(startIndex:endIndex);

%% LOAD PROVIDED LABELS (GROUND TRUTH) AND FILTER TO ROI
providedLabelsFull = importLabels(labelPath, fs);
ROIend = ROIstart + ROIlength;
providedLabelsROI = providedLabelsFull(arrayfun(@(x) (x.StartTime >= ROIstart && x.EndTime <= ROIend), providedLabelsFull));
tempProvidedFile = fullfile(tempdir, "provided_labels_ROI.txt");
saveDetectedLabels(providedLabelsROI, tempProvidedFile);

%% OPTIMIZATION LOOP OVER PSD AND ADAPTIVE THRESHOLD PARAMETERS WITH NOISE TRACKING
bestF1 = -Inf;
bestParams = struct('segmentLength',NaN, 'overlapFactor',NaN, 'maWindow',NaN, 'localWindow',NaN, 'k',NaN);
results = [];  % To store candidate parameter combinations and F1 scores

fprintf('Optimizing PSD and Adaptive Threshold Parameters with Noise Tracking on ROI...\n');
for segLen = candidateSegmentLengths
    currentWindow = hamming(segLen);
    for ov = candidateOverlapFactors
        for maW = candidateMaWindows
            % Compute spectrogram using current PSD parameters
            [S, F, T_spec] = spectrogram(xROI, currentWindow, round(segLen * ov), segLen, fs);
            freqMask = (F >= fcutMin) & (F <= fcutMax);
            powerEnvelope = sum(abs(S(freqMask, :)).^2, 1);
            powerEnvelope = powerEnvelope / max(powerEnvelope);  % Global normalization
            powerEnvelope = smoothdata(powerEnvelope, 'movmean', maW);  % Smoothing
            
            % Noise tracking: Estimate noise floor via a moving minimum filter
            noiseFloor = movmin(powerEnvelope, noiseWindow);
            % Compute effective envelope (subtract noise floor; clip negatives)
            effectiveEnvelope = powerEnvelope - noiseFloor;
            effectiveEnvelope(effectiveEnvelope < 0) = 0;
            
            % Loop over candidate adaptive threshold parameters
            for lw = candidateLocalWindows
                localMean = movmean(effectiveEnvelope, lw);
                localStd  = movstd(effectiveEnvelope, lw);
                for currentK = candidateK
                    adaptiveThreshold = localMean + currentK * localStd;
                    % Determine binary detections using the adaptive threshold on effective envelope
                    binaryDetections = effectiveEnvelope > adaptiveThreshold;
                    
                    % Group sequential detections into events
                    groupedDetections = diff([0, binaryDetections, 0]);
                    eventStarts = find(groupedDetections == 1);
                    eventEnds = find(groupedDetections == -1) - 1;
                    
                    % Convert spectrogram time indices to absolute time
                    eventTimesStart = T_spec(eventStarts) + ROIstart;
                    eventTimesEnd = T_spec(eventEnds) + ROIstart;
                    
                    nEvents = length(eventTimesStart);
                    detectedLabels = struct('StartTime', cell(nEvents,1), 'EndTime', cell(nEvents,1));
                    for i = 1:nEvents
                        detectedLabels(i).StartTime = eventTimesStart(i);
                        detectedLabels(i).EndTime = eventTimesEnd(i);
                    end
                    
                    % Save detected labels to temporary file for evaluation
                    tempDetectedFile = fullfile(tempdir, "detected_labels.txt");
                    saveDetectedLabels(detectedLabels, tempDetectedFile);
                    
                    % Evaluate detection performance using compareLabels
                    stats = compareLabels(tempProvidedFile, tempDetectedFile, fs);
                    F1 = stats.F1Score;
                    
                    results = [results; segLen, ov, maW, lw, currentK, F1]; %#ok<AGROW>
                    fprintf('segLen=%5d, ov=%.2f, maW=%d, localWin=%3d, k=%.2f --> F1=%.3f\n',...
                        segLen, ov, maW, lw, currentK, F1);
                    
                    if F1 > bestF1
                        bestF1 = F1;
                        bestParams.segmentLength = segLen;
                        bestParams.overlapFactor = ov;
                        bestParams.maWindow = maW;
                        bestParams.localWindow = lw;
                        bestParams.k = currentK;
                    end
                end
            end
        end
    end
end

fprintf('\nOptimal Parameters:\n');
fprintf('Segment Length: %d samples\n', bestParams.segmentLength);
fprintf('Overlap Factor: %.2f\n', bestParams.overlapFactor);
fprintf('Moving Average Window: %d samples\n', bestParams.maWindow);
fprintf('Adaptive Local Window: %d samples\n', bestParams.localWindow);
fprintf('k Factor: %.2f\n', bestParams.k);
fprintf('Best F1-Score: %.3f\n', bestF1);

%% FINAL DETECTION USING OPTIMAL PARAMETERS WITH NOISE TRACKING ON ROI
bestSegLen = bestParams.segmentLength;
bestOv = bestParams.overlapFactor;
bestMaW = bestParams.maWindow;
currentWindow = hamming(bestSegLen);
[S, F, T_spec] = spectrogram(xROI, currentWindow, round(bestSegLen * bestOv), bestSegLen, fs);
freqMask = (F >= fcutMin) & (F <= fcutMax);
powerEnvelope = sum(abs(S(freqMask, :)).^2, 1);
powerEnvelope = powerEnvelope / max(powerEnvelope);
powerEnvelope = smoothdata(powerEnvelope, 'movmean', bestMaW);

% Noise tracking: estimate noise floor
noiseFloor = movmin(powerEnvelope, noiseWindow);
effectiveEnvelope = powerEnvelope - noiseFloor;
effectiveEnvelope(effectiveEnvelope < 0) = 0;

% Compute optimal adaptive threshold using best adaptive parameters
bestLocalWindow = bestParams.localWindow;
bestK = bestParams.k;
localMean = movmean(effectiveEnvelope, bestLocalWindow);
localStd  = movstd(effectiveEnvelope, bestLocalWindow);
optimalThreshold = localMean + bestK * localStd;

binaryDetections = effectiveEnvelope > optimalThreshold;
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

fprintf('\nFinal Detection Performance with Optimal Parameters & Noise Tracking:\n');
fprintf('True Positives: %d\n', finalStats.TruePositives);
fprintf('False Positives: %d\n', finalStats.FalsePositives);
fprintf('False Negatives: %d\n', finalStats.FalseNegatives);
fprintf('Precision: %.3f\n', finalStats.Precision);
fprintf('Recall: %.3f\n', finalStats.Recall);
fprintf('F1-Score: %.3f\n', finalStats.F1Score);

%% OUTPUT: Save final detection results to file.
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

%% HELPER FUNCTION: saveDetectedLabels
function saveDetectedLabels(labels, filePath)
    fid = fopen(filePath, 'w');
    for i = 1:length(labels)
        fprintf(fid, '%.6f\t%.6f\tdetected\n', labels(i).StartTime, labels(i).EndTime);
        fprintf(fid, '0\t0\t\n');
    end
    fclose(fid);
end