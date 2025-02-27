%clc; close all; clear;

%id = 1;
id = id+1;
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
%% PARAMETERS (Best-Fit)
fcutMin       = 40000;    % Minimum frequency cutoff (Hz)
fcutMax       = 120000;   % Maximum frequency cutoff (Hz)
fs            = 250000;   % Sampling frequency (Hz)
ROIstart      = 60;       % ROI start time (s)
ROIlength     = 10;       % ROI length (s)
runWholeSignal = true;    % Set true to process the whole signal

% Boolean flag to control plotting
plotDetection = false;

% Best PSD parameters
segmentLength = 8192;    % Optimal segment length for Welch PSD
overlapFactor = 0.59;    % Optimal overlap factor (fraction)
maWindow      = 3;       % Optimal moving average window for smoothing

% Best Noise Tracking & Adaptive Threshold Parameters
noiseWindow   = 240;     % Optimal noise window (samples) for noise floor estimation
localWindow   = 194;     % Optimal adaptive local window (samples) for computing local mean/std
k             = 0.023;   % Optimal scaling factor for adaptive threshold
w             = 0.994;   % Optimal weight for local SNR in adaptive thresholding

% Post-Processing Parameters
minDuration = 0.0005;      % Minimum duration of an event to keep (seconds; 0.5 ms)
maxGap      = 0.001;       % Maximum gap between events to merge (seconds; 1 ms)
minEffectivePower = 0.0001;  % Minimum average effective envelope power for an event

%% File Paths
basePath  = "/Users/gazda/Documents/CTU/Masters/Masters thesis/";
datapath  = fullfile(basePath, "data");
audioPath = fullfile(datapath, "usv_audio/denoise", strcat(audioFile, "_denoise",".wav"));
labelPath = fullfile(datapath, "labels", strcat(audioFile, "-IVojt.txt"));
outputPath = fullfile(datapath, strcat(audioFile, "_detected.txt"));

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
    ROIstart = 0;               % Whole signal starts at 0
    ROIlength = t(end);         % ROI length is full duration
else
    startIndex = round(ROIstart * fs) + 1;
    endIndex   = round((ROIstart + ROIlength) * fs);
    xROI = x(startIndex:endIndex);
    tROI = t(startIndex:endIndex);
end

%% LOAD PROVIDED LABELS (GROUND TRUTH) AND FILTER TO ROI
providedLabelsFull = importLabels(labelPath, fs);
if runWholeSignal
    providedLabelsROI = providedLabelsFull;
else
    ROIend = ROIstart + ROIlength;
    providedLabelsROI = providedLabelsFull(arrayfun(@(x) (x.StartTime >= ROIstart && x.EndTime <= ROIend), providedLabelsFull));
end
tempProvidedFile = fullfile(tempdir, "provided_labels_ROI.txt");
saveDetectedLabels(providedLabelsROI, tempProvidedFile);

%% PSD ESTIMATION
nfft = segmentLength;
window = hamming(segmentLength);
[S, F, T_spec] = spectrogram(xROI, window, round(segmentLength * overlapFactor), nfft, fs);
freqMask = (F >= fcutMin) & (F <= fcutMax);
powerEnvelope = sum(abs(S(freqMask, :)).^2, 1);
powerEnvelope = powerEnvelope / max(powerEnvelope);
powerEnvelope = smoothdata(powerEnvelope, 'movmean', maWindow);

%% NOISE TRACKING & EFFECTIVE ENVELOPE
% Estimate noise floor using a moving minimum filter
noiseFloor = movmin(powerEnvelope, noiseWindow);
effectiveEnvelope = powerEnvelope - noiseFloor;
effectiveEnvelope(effectiveEnvelope < 0) = 0;

%% LOCAL SNR CALCULATION
localSNR = effectiveEnvelope ./ (noiseFloor + eps);
localSNR = min(localSNR, 10);

%% ADAPTIVE THRESHOLDING
localMean = movmean(effectiveEnvelope, localWindow);
localStd  = movstd(effectiveEnvelope, localWindow);
optimalThreshold = (localMean + k * localStd) ./ (1 + w * localSNR);

%% DETECTION
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

%% POST-PROCESSING
% Remove events shorter than minDuration and with very low effective power.
validEvents = [];
for i = 1:length(detectedLabels)
    duration = detectedLabels(i).EndTime - detectedLabels(i).StartTime;
    % Get indices corresponding to the event in T_spec
    idx = find((T_spec + ROIstart) >= detectedLabels(i).StartTime & (T_spec + ROIstart) <= detectedLabels(i).EndTime);
    avgPower = mean(effectiveEnvelope(idx));
    if duration >= minDuration && avgPower >= minEffectivePower
        validEvents = [validEvents; detectedLabels(i)]; %#ok<AGROW>
    end
end
detectedLabels = validEvents;

% Merge events if the gap between them is less than maxGap
if length(detectedLabels) > 1
    mergedEvents = detectedLabels(1);
    for i = 2:length(detectedLabels)
        gap = detectedLabels(i).StartTime - mergedEvents(end).EndTime;
        if gap < maxGap
            mergedEvents(end).EndTime = detectedLabels(i).EndTime;
        else
            mergedEvents = [mergedEvents; detectedLabels(i)]; %#ok<AGROW>
        end
    end
    detectedLabels = mergedEvents;
end
nEvents = length(detectedLabels);

%% STATISTICAL ANALYSIS
tempDetectedFile = fullfile(tempdir, "detected_labels.txt");
saveDetectedLabels(detectedLabels, tempDetectedFile);
stats = compareLabels(tempProvidedFile, tempDetectedFile, fs);

% fprintf("Statistical Analysis (Midpoint Criterion):\n");
% fprintf("True Positives: %d\n", stats.TruePositives);
% fprintf("False Positives: %d\n", stats.FalsePositives);
% fprintf("False Negatives: %d\n", stats.FalseNegatives);
% fprintf("Precision: %.3f\n", stats.Precision);
% fprintf("Recall: %.3f\n", stats.Recall);
% fprintf("F1-Score: %.3f\n", stats.F1Score);

%% PLOTTING (if enabled)
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
    xlabel("Time (s)"); ylabel("Amplitude");
    grid on;
    
    % Plot spectrogram
    nexttile;
    imagesc(T_spec + ROIstart, F/1000, 10*log10(abs(S)));
    axis xy; colormap jet; colorbar;
    title("Spectrogram");
    xlabel("Time (s)"); ylabel("Frequency (kHz)");
    
    % Plot smoothed power envelope and adaptive threshold
    nexttile;
    plot(T_spec + ROIstart, powerEnvelope, 'LineWidth', 1.5);
    hold on;
    plot(T_spec + ROIstart, optimalThreshold, 'k--', 'LineWidth', 1.5);
    hold off;
    title("Smoothed Power Envelope & Adaptive Threshold");
    xlabel("Time (s)"); ylabel("Normalized Power");
    grid on;
    
    % Plot binary detection results
    nexttile;
    stem(T_spec + ROIstart, binaryDetections, 'r');
    title("Binary Detections");
    xlabel("Time (s)"); ylabel("Detection (1/0)");
    grid on;
    
    % Plot detected events over raw signal
    nexttile;
    plot(tROI, xROI);
    hold on;
    for i = 1:nEvents
        xline(detectedLabels(i).StartTime, 'b--', 'LineWidth', 1.5);
        xline(detectedLabels(i).EndTime, 'g--', 'LineWidth', 1.5);
    end
    hold off;
    title("Detected Events Over Signal");
    xlabel("Time (s)"); ylabel("Amplitude");
    grid on;
end

%% OUTPUT
% fprintf("Detection complete. Total events detected: %d\n", nEvents);
% for i = 1:nEvents
%     fprintf("Event %d: Start Time = %.3f s, End Time = %.3f s\n", i, detectedLabels(i).StartTime, detectedLabels(i).EndTime);
% end
% 
% fileID = fopen(outputPath, 'w');
% fprintf(fileID, "Detection complete. Total events detected: %d\n", nEvents);
% for i = 1:nEvents
%     fprintf(fileID, "Event %d: Start Time = %.3f s, End Time = %.3f s\n", i, detectedLabels(i).StartTime, detectedLabels(i).EndTime);
% end
% fclose(fileID);
% fprintf("Detection results saved to: %s\n", outputPath);


saveDetectedLabels(detectedLabels,outputPath);
%% HELPER FUNCTION: saveDetectedLabels
function saveDetectedLabels(labels, filePath)
    fid = fopen(filePath, 'w');
    for i = 1:length(labels)
        fprintf(fid, '%.6f\t%.6f\td\n', labels(i).StartTime, labels(i).EndTime);
        fprintf(fid, '\\\t0.000000\t0.000000\n');
    end
    fclose(fid);
end