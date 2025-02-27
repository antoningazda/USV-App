%% optimizePSD_withThreshold.m
% This script optimizes the PSD estimation parameters and the detection threshold 
% for your USV detector. It searches over candidate segment lengths, overlap factors,
% moving average window sizes, and detection thresholds.
% The performance is evaluated using the F1‐score computed via compareLabels.m.
% Detected events (obtained via power envelope thresholding) are compared to provided labels.

clc; clear; close all;

%% Fixed Parameters
fcutMin   = 40000;        % Minimum frequency cutoff (Hz)
fcutMax   = 120000;       % Maximum frequency cutoff (Hz)
fs        = 250000;        % Sampling frequency (Hz)
ROIstart  = 100;           % ROI start time (s)
ROIlength = 20;            % ROI length (s)

% %
% Optimal Parameters:
% Segment Length: 16384 samples
% Overlap Factor: 0.60
% Moving Average Window: 3 samples
% Detection Threshold: 0.040
% Best F1-Score: 0.839

%% Candidate Parameters
% PSD parameters
segmentLengths = [1024, 2048, 4096, 8192, 16384];   % Candidate segment lengths (FFT lengths)
overlapFactors = [0.25, 0.4, 0.5, 0.6, 0.75];         % Candidate overlap factors (fraction)
maWindows      = [1, 2, 3, 4, 5];                     % Moving average window sizes (number of power envelope samples)

% Detection threshold candidates
thresholds = 0.01:0.01:0.10;                         % Candidate thresholds

%% File Paths
basePath  = "/Users/gazda/Documents/CTU/Masters/Masters thesis/";
dataPath  = fullfile(basePath, "data");
audioFile = "LPS-SI2homo-mH02-I04-USV";
audioPath = fullfile(dataPath, "usv_audio", audioFile + ".wav");
labelPath = fullfile(dataPath, "labels", audioFile + "-IVojt.txt");

%% Load Audio and Extract ROI
[x, fs_audio] = audioread(audioPath);
if fs_audio ~= fs
    x = resample(x, fs, fs_audio);
end
t = (0:length(x)-1) / fs;
x = x - mean(x);           % Remove DC offset
x = x / max(abs(x));       % Normalize

startIndex = round(ROIstart * fs) + 1;
endIndex   = round((ROIstart + ROIlength) * fs);
xROI = x(startIndex:endIndex);
tROI = t(startIndex:endIndex);

%% Load Provided (Ground Truth) Labels and Filter to ROI
providedLabelsFull = importLabels(labelPath, fs);
ROIend = ROIstart + ROIlength;
providedLabelsROI = providedLabelsFull(arrayfun(@(x) (x.StartTime >= ROIstart && x.EndTime <= ROIend), providedLabelsFull));

% Save provided labels to temporary file (format expected by compareLabels)
tempProvidedFile = fullfile(tempdir, "provided_labels_ROI.txt");
saveDetectedLabels(providedLabelsROI, tempProvidedFile);

%% Optimization Loop Over PSD and Threshold Parameters
bestF1 = -Inf;
bestParams = struct('segmentLength', NaN, 'overlapFactor', NaN, 'maWindow', NaN, 'threshold', NaN);
results = [];  % To store parameter combinations and their F1 scores

fprintf('Optimizing PSD parameters and detection threshold...\n');
for segLen = segmentLengths
    window = hamming(segLen);  % Use Hamming window for the current segment length
    for ov = overlapFactors
        for maW = maWindows
            for thr = thresholds
                % Compute spectrogram over ROI with current PSD parameters
                [S, F, T_spec] = spectrogram(xROI, window, round(segLen * ov), segLen, fs);
                
                % Restrict analysis to the frequency band of interest
                freqMask = (F >= fcutMin) & (F <= fcutMax);
                % Compute power envelope as the sum of squared magnitudes over the freq band
                powerEnvelope = sum(abs(S(freqMask, :)).^2, 1);
                powerEnvelope = powerEnvelope / max(powerEnvelope);  % Normalize
                
                % Apply moving average smoothing
                powerEnvelope = smoothdata(powerEnvelope, 'movmean', maW);
                
                % Detection: threshold the power envelope with current threshold value
                binaryDetections = powerEnvelope > thr;
                groupedDetections = diff([0, binaryDetections, 0]);
                eventStarts = find(groupedDetections == 1);
                eventEnds   = find(groupedDetections == -1) - 1;
                
                % Convert spectrogram time indices to absolute time (adding ROI offset)
                eventTimesStart = T_spec(eventStarts) + ROIstart;
                eventTimesEnd   = T_spec(eventEnds) + ROIstart;
                
                % Build a struct array of detected events
                nEvents = length(eventTimesStart);
                detectedLabels = struct('StartTime', cell(nEvents, 1), 'EndTime', cell(nEvents, 1));
                for i = 1:nEvents
                    detectedLabels(i).StartTime = eventTimesStart(i);
                    detectedLabels(i).EndTime   = eventTimesEnd(i);
                end
                
                % Save detected labels to a temporary file
                tempDetectedFile = fullfile(tempdir, "detected_labels.txt");
                saveDetectedLabels(detectedLabels, tempDetectedFile);
                
                % Evaluate detection performance using the provided compareLabels function
                stats = compareLabels(tempProvidedFile, tempDetectedFile, fs);
                F1 = stats.F1Score;
                
                % Store the current parameters and F1-score
                results = [results; segLen, ov, maW, thr, F1];  %#ok<AGROW>
                fprintf('segLen=%5d, ov=%.2f, maWindow=%d, thr=%.3f, F1=%.3f\n', segLen, ov, maW, thr, F1);
                
                % Update best parameters if F1 improves
                if F1 > bestF1
                    bestF1 = F1;
                    bestParams.segmentLength = segLen;
                    bestParams.overlapFactor = ov;
                    bestParams.maWindow = maW;
                    bestParams.threshold = thr;
                end
            end
        end
    end
end

fprintf('\nOptimal Parameters:\n');
fprintf('Segment Length: %d samples\n', bestParams.segmentLength);
fprintf('Overlap Factor: %.2f\n', bestParams.overlapFactor);
fprintf('Moving Average Window: %d samples\n', bestParams.maWindow);
fprintf('Detection Threshold: %.3f\n', bestParams.threshold);
fprintf('Best F1-Score: %.3f\n', bestF1);

%% Helper Function: saveDetectedLabels
function saveDetectedLabels(labels, filePath)
    % Saves labels to a text file in the format expected by importLabels.
    fid = fopen(filePath, 'w');
    for i = 1:length(labels)
        fprintf(fid, '%.6f\t%.6f\tdetected\n', labels(i).StartTime, labels(i).EndTime);
        fprintf(fid, '0\t0\t\n');
    end
    fclose(fid);
end
