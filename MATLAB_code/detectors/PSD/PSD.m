clc; close all; clear;

%% FIXED PARAMETERS (from previous optimization)
fcutMin       = 40000;        % Minimum frequency cutoff (Hz)
fcutMax       = 120000;       % Maximum frequency cutoff (Hz)
segmentLength = 16384;        % Optimal segment length for Welch PSD
overlapFactor = 0.60;         % Optimal overlap factor (fraction)
maWindow      = 3;            % Optimal moving average window size (samples)
fs            = 250000;       % Sampling frequency (Hz)
ROIstart      = 100;          % Start time of Region of Interest (ROI) in seconds
ROIlength     = 20;           % Length of ROI in seconds

% Set to true to process the whole signal instead of the ROI
runWholeSignal = false;
plotDetection  = false;         % Set to true to see the final detection plots

%% Candidate Adaptive Threshold Parameters
% We'll optimize the parameters of the adaptive threshold:
% adaptiveThreshold = localMean + k * localStd
candidateLocalWindow = [100, 200, 300, 500];  % window sizes (in samples)
candidateK           = [0.3, 0.4, 0.5, 0.6, 0.7];  % scaling factors

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
x = x - mean(x);       % Remove DC offset
x = x / max(abs(x));   % Normalize audio

%% SELECT SIGNAL PORTION: ROI or Whole Signal
if runWholeSignal
    xROI = x;
    tROI = t;
    ROIstart = 0;               % Whole signal starts at 0
    ROIlength = t(end);         % ROI length is the full duration
else
    startIndex = round(ROIstart * fs) + 1;
    endIndex   = round((ROIstart + ROIlength) * fs);
    xROI = x(startIndex:endIndex);
    tROI = t(startIndex:endIndex);
end

%% PREPARE PSD ESTIMATION
nfft = segmentLength;              % FFT length
window = hamming(segmentLength);   % Hamming window

% Compute spectrogram over selected portion
[S, F, T_spec] = spectrogram(xROI, window, round(segmentLength * overlapFactor), nfft, fs);

% Compute power envelope over the frequency band of interest
freqMask = (F >= fcutMin) & (F <= fcutMax);
powerEnvelope_orig = sum(abs(S(freqMask, :)).^2, 1);
powerEnvelope_orig = powerEnvelope_orig / max(powerEnvelope_orig);  % Normalize
% Apply fixed smoothing (maWindow)
powerEnvelope_orig = smoothdata(powerEnvelope_orig, 'movmean', maWindow);

%% Prepare Provided Labels (Ground Truth)
providedLabelsFull = importLabels(labelPath, fs);
if runWholeSignal
    providedLabelsROI = providedLabelsFull;
else
    ROIend = ROIstart + ROIlength;
    providedLabelsROI = providedLabelsFull(arrayfun(@(x) (x.StartTime >= ROIstart && x.EndTime <= ROIend), providedLabelsFull));
end
% Save provided labels to a temporary file (for compareLabels)
tempProvidedFile = fullfile(tempdir, "provided_labels_ROI.txt");
saveDetectedLabels(providedLabelsROI, tempProvidedFile);

%% OPTIMIZATION LOOP OVER ADAPTIVE THRESHOLD PARAMETERS
bestF1 = -Inf;
bestParams = struct('localWindow', NaN, 'k', NaN);
results = [];  % To store candidate parameters and F1 scores

fprintf('Optimizing adaptive threshold parameters...\n');
for lw = candidateLocalWindow
    % Compute local mean and std over the power envelope (using candidate window size)
    localMean = movmean(powerEnvelope_orig, lw);
    localStd  = movstd(powerEnvelope_orig, lw);
    for currentK = candidateK
        % Compute adaptive threshold
        adaptiveThreshold = localMean + currentK * localStd;
        
        % Determine binary detections using adaptive threshold
        binaryDetections = powerEnvelope_orig > adaptiveThreshold;
        
        % Group sequential detections into events
        groupedDetections = diff([0, binaryDetections, 0]);
        eventStarts = find(groupedDetections == 1);
        eventEnds   = find(groupedDetections == -1) - 1;
        
        % Convert spectrogram time indices to absolute time (add ROIstart)
        eventTimesStart = T_spec(eventStarts) + ROIstart;
        eventTimesEnd   = T_spec(eventEnds) + ROIstart;
        
        % Build detectedLabels struct array
        nEvents = length(eventTimesStart);
        detectedLabels = struct('StartTime', cell(nEvents,1), 'EndTime', cell(nEvents,1));
        for i = 1:nEvents
            detectedLabels(i).StartTime = eventTimesStart(i);
            detectedLabels(i).EndTime   = eventTimesEnd(i);
        end
        
        % Save detected labels to temporary file for evaluation
        tempDetectedFile = fullfile(tempdir, "detected_labels.txt");
        saveDetectedLabels(detectedLabels, tempDetectedFile);
        
        % Evaluate detection performance using compareLabels
        stats = compareLabels(tempProvidedFile, tempDetectedFile, fs);
        F1 = stats.F1Score;
        
        % Store the current candidate parameters and F1 score
        results = [results; lw, currentK, F1]; %#ok<AGROW>
        fprintf('localWindow=%3d, k=%.2f --> F1=%.3f\n', lw, currentK, F1);
        
        % Update best parameters if F1 improves
        if F1 > bestF1
            bestF1 = F1;
            bestParams.localWindow = lw;
            bestParams.k = currentK;
        end
    end
end

fprintf('\nOptimal Adaptive Threshold Parameters:\n');
fprintf('Local Window: %d samples\n', bestParams.localWindow);
fprintf('k Factor: %.2f\n', bestParams.k);
fprintf('Best F1-Score: %.3f\n', bestF1);

%% FINAL DETECTION USING OPTIMAL ADAPTIVE PARAMETERS
% Recompute adaptive threshold with best parameters
localMean = movmean(powerEnvelope_orig, bestParams.localWindow);
localStd  = movstd(powerEnvelope_orig, bestParams.localWindow);
optimalThreshold = localMean + bestParams.k * localStd;
binaryDetections = powerEnvelope_orig > optimalThreshold;
groupedDetections = diff([0, binaryDetections, 0]);
eventStarts = find(groupedDetections == 1);
eventEnds = find(groupedDetections == -1) - 1;
eventTimesStart = T_spec(eventStarts) + ROIstart;
eventTimesEnd   = T_spec(eventEnds) + ROIstart;
nEvents = length(eventTimesStart);
detectedLabels = struct('StartTime', cell(nEvents,1), 'EndTime', cell(nEvents,1));
for i = 1:nEvents
    detectedLabels(i).StartTime = eventTimesStart(i);
    detectedLabels(i).EndTime   = eventTimesEnd(i);
end

% Save final detected labels to output file
tempDetectedFile = fullfile(tempdir, "detected_labels.txt");
saveDetectedLabels(detectedLabels, tempDetectedFile);
stats = compareLabels(tempProvidedFile, tempDetectedFile, fs);
fprintf('\nFinal Detection Performance with Optimal Adaptive Parameters:\n');
fprintf("True Positives: %d\n", stats.TruePositives);
fprintf("False Positives: %d\n", stats.FalsePositives);
fprintf("False Negatives: %d\n", stats.FalseNegatives);
fprintf("Precision: %.3f\n", stats.Precision);
fprintf("Recall: %.3f\n", stats.Recall);
fprintf("F1-Score: %.3f\n", stats.F1Score);

%% PLOTTING (Optional)
if plotDetection
    figure;
    tiledlayout(5,1);
    
    % Plot raw signal
    nexttile;
    plot(tROI, xROI);
    if runWholeSignal
        title("Full Signal");
    else
        title("ROI Signal");
    end
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
    
    % Plot smoothed power envelope and optimal adaptive threshold
    nexttile;
    plot(T_spec + ROIstart, powerEnvelope_orig, 'LineWidth', 1.5);
    hold on;
    plot(T_spec + ROIstart, optimalThreshold, 'k--', 'LineWidth', 1.5);
    hold off;
    title("Smoothed Power Envelope & Optimal Adaptive Threshold");
    xlabel("Time (s)");
    ylabel("Normalized Power");
    axis tight;
    grid on;
    
    % Plot binary detection results
    nexttile;
    stem(T_spec + ROIstart, binaryDetections, 'r');
    title("Binary Detections");
    xlabel("Time (s)");
    ylabel("Detection (1/0)");
    axis tight;
    
    % Plot detection events on raw signal
    nexttile;
    plot(tROI, xROI);
    hold on;
    for i = 1:nEvents
        xline(detectedLabels(i).StartTime, 'b--', 'LineWidth', 1.5);
    end
    hold off;
    title("Detected Events on Signal");
    xlabel("Time (s)");
    ylabel("Amplitude");
    axis tight;
end

%% OUTPUT: Save Final Detection Results to File
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
    % Save labels to a text file in the format expected by importLabels.
    fid = fopen(filePath, 'w');
    for i = 1:length(labels)
        fprintf(fid, '%.6f\t%.6f\tdetected\n', labels(i).StartTime, labels(i).EndTime);
        fprintf(fid, '0\t0\t\n');
    end
    fclose(fid);
end