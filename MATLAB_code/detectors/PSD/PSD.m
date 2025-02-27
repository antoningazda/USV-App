clc; close all; clear;

%% PARAMETERS
fcutMin = 40000;        % Minimum frequency cutoff (Hz)
fcutMax = 120000;       % Maximum frequency cutoff (Hz)
threshold = 0.040;       % Optimal detection threshold
segmentLength = 16384;   % Optimal segment length for Welch PSD
overlapFactor = 0.60;    % Optimal overlap factor (fraction)
maWindow = 3;            % Optimal moving average window size (in number of samples)
fs = 250000;             % Sampling frequency (Hz)
ROIstart = 100;          % Start time of Region of Interest (ROI) in seconds
ROIlength = 20;          % Length of ROI in seconds

%% File Paths
basePath = "/Users/gazda/Documents/CTU/Masters/Masters thesis/";
datapath = fullfile(basePath, "data");
audioFile = "LPS-SI2homo-mH02-I04-USV";
audioPath = fullfile(datapath, "usv_audio", audioFile + ".wav");
labelPath = fullfile(datapath, "labels", audioFile + "-IVojt.txt");
outputPath = fullfile(datapath, audioFile + "_detected.txt");

%% LOAD AUDIO
[x, fs] = audioread(audioPath);
t = (0:length(x)-1) / fs;      % Time vector
x = x - mean(x);             % Remove DC offset
x = x / max(abs(x));         % Normalize audio

% Extract ROI from the full signal
startIndex = round(ROIstart * fs) + 1;
endIndex = round((ROIstart + ROIlength) * fs);
xROI = x(startIndex:endIndex);
tROI = t(startIndex:endIndex);

%% PSD ESTIMATION
% Create Hamming window of optimal segment length
nfft = segmentLength;
window = hamming(segmentLength);

% Compute spectrogram over ROI
% The number of overlapping samples is round(segmentLength * overlapFactor)
[S, F, T_spec] = spectrogram(xROI, window, round(segmentLength * overlapFactor), nfft, fs);

% Select frequencies of interest and calculate power envelope
freqMask = (F >= fcutMin) & (F <= fcutMax);
powerEnvelope = sum(abs(S(freqMask, :)).^2, 1);
powerEnvelope = powerEnvelope / max(powerEnvelope);  % Normalize

% Apply moving average smoothing to the power envelope using optimal window
powerEnvelope = smoothdata(powerEnvelope, 'movmean', maWindow);

%% DETECTION
% Binary detection using thresholding of the smoothed power envelope
binaryDetections = powerEnvelope > threshold;

% Group sequential detections into events by taking the difference
groupedDetections = diff([0, binaryDetections, 0]);
eventStarts = find(groupedDetections == 1);    % Indices where events start
eventEnds = find(groupedDetections == -1) - 1;   % Indices where events end

% Convert spectrogram time indices to absolute time (by adding ROIstart)
eventTimesStart = T_spec(eventStarts) + ROIstart;
eventTimesEnd   = T_spec(eventEnds) + ROIstart;

% Build a struct array of detected events with start and end times
nEvents = length(eventTimesStart);
detectedLabels = struct('StartTime', cell(nEvents,1), 'EndTime', cell(nEvents,1));
for i = 1:nEvents
    detectedLabels(i).StartTime = eventTimesStart(i);
    detectedLabels(i).EndTime = eventTimesEnd(i);
end

%% STATISTICAL ANALYSIS
% Save detected labels to a temporary file (format expected by importLabels)
tempDetectedFile = fullfile(tempdir, "detected_labels.txt");
saveDetectedLabels(detectedLabels, tempDetectedFile);

% Load provided (ground truth) labels and filter them to the ROI.
providedLabelsFull = importLabels(labelPath, fs);
ROIend = ROIstart + ROIlength;
providedLabelsROI = providedLabelsFull(arrayfun(@(x) (x.StartTime >= ROIstart && x.EndTime <= ROIend), providedLabelsFull));

% Save the ROI-filtered provided labels to a temporary file.
tempProvidedFile = fullfile(tempdir, "provided_labels_ROI.txt");
saveDetectedLabels(providedLabelsROI, tempProvidedFile);

% Evaluate detection performance using the provided compareLabels function.
stats = compareLabels(tempProvidedFile, tempDetectedFile, fs);

fprintf("Statistical Analysis (Midpoint Criterion):\n");
fprintf("True Positives: %d\n", stats.TruePositives);
fprintf("False Positives: %d\n", stats.FalsePositives);
fprintf("False Negatives: %d\n", stats.FalseNegatives);
fprintf("Precision: %.3f\n", stats.Precision);
fprintf("Recall: %.3f\n", stats.Recall);
fprintf("F1-Score: %.3f\n", stats.F1Score);

%% PLOTTING
figure;
tiledlayout(4,1);

% Plot raw ROI signal
nexttile;
plot(tROI, xROI);
title("ROI Signal");
xlabel("Time (s)");
ylabel("Amplitude");
axis tight;

% Plot spectrogram
nexttile;
imagesc(T_spec + ROIstart, F/1000, 10*log10(abs(S)));
axis xy;
colormap jet;
ylim([fcutMin, fcutMax] / 1000);
colorbar;
title("Spectrogram");
xlabel("Time (s)");
ylabel("Frequency (kHz)");

% Plot smoothed power envelope
nexttile;
plot(T_spec + ROIstart, powerEnvelope, 'LineWidth', 1.5);
title("Smoothed Time-Varying Power");
xlabel("Time (s)");
ylabel("Normalized Power");
axis tight;
grid on;

% Plot detection results (binary detections)
nexttile;
stem(T_spec + ROIstart, binaryDetections, 'r');
hold on;
for i = 1:nEvents
    xline(detectedLabels(i).StartTime, 'b--', 'LineWidth', 1.5);
end
title("Detection Results");
xlabel("Time (s)");
ylabel("Detection (1/0)");
axis tight;

%% OUTPUT
fprintf("Detection complete. Total events detected: %d\n", nEvents);
for i = 1:nEvents
    fprintf("Event %d: Start Time = %.3f s, End Time = %.3f s\n", i, detectedLabels(i).StartTime, detectedLabels(i).EndTime);
end

% Save detection results to output file.
fileID = fopen(outputPath, 'w');
fprintf(fileID, "Detection complete. Total events detected: %d\n", nEvents);
for i = 1:nEvents
    fprintf(fileID, "Event %d: Start Time = %.3f s, End Time = %.3f s\n", i, detectedLabels(i).StartTime, detectedLabels(i).EndTime);
end
fclose(fileID);
fprintf("Detection results saved to: %s\n", outputPath);

%% HELPER FUNCTION: saveDetectedLabels
function saveDetectedLabels(labels, filePath)
    % Save labels to a text file in the format expected by importLabels.
    fid = fopen(filePath, 'w');
    for i = 1:length(labels)
        fprintf(fid, '%.6f\t%.6f\tdetected\n', labels(i).StartTime, labels(i).EndTime);
        fprintf(fid, '0\t0\t\n');
    end
    fclose(fid);
end