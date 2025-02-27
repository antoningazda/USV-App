clc; close all; clear;

%% FIXED PARAMETERS (for candidate search)
fcutMin       = 40000;   % Minimum frequency cutoff (Hz)
fcutMax       = 120000;  % Maximum frequency cutoff (Hz)
fs            = 250000;  % Sampling frequency (Hz)
ROIstart      = 100;     % ROI start time (s)
ROIlength     = 20;      % ROI length (s)
runWholeSignal = false;   % Set true to process the entire signal
plotDetection  = false;

%
% Final Detection Performance with Optimal Parameters:
% True Positives: 31
% False Positives: 4
% False Negatives: 28
% Precision: 0.886
% Recall: 0.525
% F1-Score: 0.660

%% Candidate PSD Parameters
candidateSegmentLengths = [8192, 16384]; % FFT lengths
candidateOverlapFactors = [0.6, 0.75];       % Overlap fractions
candidateMaWindows      = [1, 2, 3, 4, 5];                    % Smoothing window for power envelope

%% Candidate Adaptive Threshold Parameters
% Adaptive threshold is computed as: localMean + k * localStd
candidateLocalWindows = [25, 50, 75, 100, 200, 300, 400];  % Window sizes (in samples)
candidateK           = [0.3, 0.4, 0.5, 0.6, 0.7];  % Scaling factors

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

%% SELECT SIGNAL PORTION: ROI or Whole Signal
if runWholeSignal
    xROI = x;
    tROI = t;
    ROIstart = 0;               % Whole signal: start at 0
    ROIlength = t(end);         % Full duration
else
    startIndex = round(ROIstart * fs) + 1;
    endIndex = round((ROIstart + ROIlength) * fs);
    xROI = x(startIndex:endIndex);
    tROI = t(startIndex:endIndex);
end

%% Load Provided (Ground Truth) Labels and Filter to ROI if needed
providedLabelsFull = importLabels(labelPath, fs);
if runWholeSignal
    providedLabelsROI = providedLabelsFull;
else
    ROIend = ROIstart + ROIlength;
    providedLabelsROI = providedLabelsFull(arrayfun(@(x) (x.StartTime >= ROIstart && x.EndTime <= ROIend), providedLabelsFull));
end
% Save provided labels to temporary file for evaluation.
tempProvidedFile = fullfile(tempdir, "provided_labels_ROI.txt");
saveDetectedLabels(providedLabelsROI, tempProvidedFile);

%% OPTIMIZATION LOOP
bestF1 = -Inf;
bestParams = struct('segmentLength',NaN, 'overlapFactor',NaN, 'maWindow',NaN, 'localWindow',NaN, 'k',NaN);
results = [];  % To store each candidate combination and its F1 score

fprintf('Optimizing PSD and Adaptive Threshold Parameters...\n');
for segLen = candidateSegmentLengths
    % Prepare window for current segment length
    currentWindow = hamming(segLen);
    for ov = candidateOverlapFactors
        for maW = candidateMaWindows
            % Compute spectrogram with current PSD parameters
            [S, F, T_spec] = spectrogram(xROI, currentWindow, round(segLen * ov), segLen, fs);
            freqMask = (F >= fcutMin) & (F <= fcutMax);
            powerEnvelope = sum(abs(S(freqMask, :)).^2, 1);
            powerEnvelope = powerEnvelope / max(powerEnvelope);  % Global normalization
            powerEnvelope = smoothdata(powerEnvelope, 'movmean', maW);  % Apply smoothing
            
            % Now loop over adaptive threshold parameters
            for lw = candidateLocalWindows
                localMean = movmean(powerEnvelope, lw);
                localStd  = movstd(powerEnvelope, lw);
                for currentK = candidateK
                    % Compute adaptive threshold
                    adaptiveThreshold = localMean + currentK * localStd;
                    
                    % Determine binary detections using adaptive threshold
                    binaryDetections = powerEnvelope > adaptiveThreshold;
                    
                    % Group sequential detections into events
                    groupedDetections = diff([0, binaryDetections, 0]);
                    eventStarts = find(groupedDetections == 1);
                    eventEnds   = find(groupedDetections == -1) - 1;
                    
                    % Convert spectrogram time indices to absolute time
                    eventTimesStart = T_spec(eventStarts) + ROIstart;
                    eventTimesEnd   = T_spec(eventEnds) + ROIstart;
                    
                    nEvents = length(eventTimesStart);
                    detectedLabels = struct('StartTime', cell(nEvents,1), 'EndTime', cell(nEvents,1));
                    for i = 1:nEvents
                        detectedLabels(i).StartTime = eventTimesStart(i);
                        detectedLabels(i).EndTime   = eventTimesEnd(i);
                    end
                    
                    % Save detected labels to temporary file
                    tempDetectedFile = fullfile(tempdir, "detected_labels.txt");
                    saveDetectedLabels(detectedLabels, tempDetectedFile);
                    
                    % Evaluate using compareLabels
                    stats = compareLabels(tempProvidedFile, tempDetectedFile, fs);
                    F1 = stats.F1Score;
                    
                    % Store results and update best if improved
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

%% FINAL DETECTION USING BEST PARAMETERS
% Recompute PSD using best PSD parameters
bestSegLen = bestParams.segmentLength;
bestOv = bestParams.overlapFactor;
bestMaW = bestParams.maWindow;
currentWindow = hamming(bestSegLen);
[S, F, T_spec] = spectrogram(xROI, currentWindow, round(bestSegLen * bestOv), bestSegLen, fs);
freqMask = (F >= fcutMin) & (F <= fcutMax);
powerEnvelope = sum(abs(S(freqMask, :)).^2, 1);
powerEnvelope = powerEnvelope / max(powerEnvelope);
powerEnvelope = smoothdata(powerEnvelope, 'movmean', bestMaW);

% Compute optimal adaptive threshold using best adaptive parameters
bestLocalWindow = bestParams.localWindow;
bestK = bestParams.k;
localMean = movmean(powerEnvelope, bestLocalWindow);
localStd  = movstd(powerEnvelope, bestLocalWindow);
optimalThreshold = localMean + bestK * localStd;

binaryDetections = powerEnvelope > optimalThreshold;
groupedDetections = diff([0, binaryDetections, 0]);
eventStarts = find(groupedDetections == 1);
eventEnds = find(groupedDetections == -1) - 1;
eventTimesStart = T_spec(eventStarts) + ROIstart;
eventTimesEnd = T_spec(eventEnds) + ROIstart;
nEvents = length(eventTimesStart);
detectedLabels = struct('StartTime', cell(nEvents,1), 'EndTime', cell(nEvents,1));
for i = 1:nEvents
    detectedLabels(i).StartTime = eventTimesStart(i);
    detectedLabels(i).EndTime   = eventTimesEnd(i);
end

% Save final detections and compute final statistics.
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

%% (Optional) PLOTTING
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
    plot(T_spec + ROIstart, powerEnvelope, 'LineWidth', 1.5);
    hold on;
    plot(T_spec + ROIstart, optimalThreshold, 'k--', 'LineWidth', 1.5);
    hold off;
    title("Smoothed Power Envelope & Optimal Adaptive Threshold");
    xlabel("Time (s)");
    ylabel("Normalized Power");
    axis tight;
    grid on;
    
    % Plot binary detections
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
    % Save labels to a text file in the format expected by importLabels.
    fid = fopen(filePath, 'w');
    for i = 1:length(labels)
        fprintf(fid, '%.6f\t%.6f\tdetected\n', labels(i).StartTime, labels(i).EndTime);
        fprintf(fid, '0\t0\t\n');
    end
    fclose(fid);
end