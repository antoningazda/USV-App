%clc; close all; clear;

%% =============== SELECT AUDIO FILE ===============
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

%% =============== PARAMETERS (Best-Fit) ===============
fcutMin       = 40000;    % Minimum frequency cutoff (Hz)
fcutMax       = 120000;   % Maximum frequency cutoff (Hz)
fs            = 250000;   % Sampling frequency (Hz)
ROIstart      = 50;       % ROI start time (s)
ROIlength     = 20;       % ROI length (s)
runWholeSignal = true;    % Process whole signal if true

% Flag to control plotting
plotDetection = false;  % set true to enable plotting

% BSCD detector parameters
wlen = 0.01;      % Window length for BSCD (seconds)
maWindow      = 5000;       % Moving average window for smoothing

% Noise tracking & adaptive threshold parameters
noiseWindow   = 256;     % Noise window for noise floor estimation
localWindow   = 256;     % Local window for computing mean and std
k             = 0.023;   % Scaling factor for adaptive threshold
w             = 0.994;   % Weight for local SNR in thresholding

% Post-Processing Parameters
minDuration = 0.005;      % Minimum event duration (seconds; 0.5 ms)
maxGap      = 0.0001;       % Maximum gap between events to merge (seconds; 1 ms)
minEffectivePower = 0.0001;  % Minimum average effective envelope power for an event

% ==================== SETTINGS ====================
testStartTime = datetime('now', 'TimeZone', 'local', 'Format', 'yyyy-MM-dd HH:mm:ss');
fprintf('Test start time: %s\n', testStartTime);

% File Paths
path = "/Users/gazda/Documents/CTU/Masters/Masters thesis/";
datapath = path + "data/";
audioPath = datapath + "usv_audio/denoise/" + audioFile + "_denoise.wav";
labelPath = datapath + "labels/" + audioFile + "-IVojt.txt";
outputPath = audioFile + "_detected.txt";
fprintf('Filename: %s\n', audioFile);

% Detector Settings (bandpass filter)
lowFreq = 40000;   % Bandpass filter lower bound (Hz)
highFreq = 120000; % Bandpass filter upper bound (Hz)

%% ==================== LOAD DATA ====================

[audioData, fs_audio] = audioread(audioPath);
% if fs_audio ~= fs
%     x = resample(x, fs, fs_audio);
% end
t = (0:length(audioData)-1) / fs;
audioData = audioData - mean(audioData);
audioData = audioData / max(abs(audioData));

%% ==================== Define Time Segment ====================
if runWholeSignal
    xROI = audioData;
    tROI = t;
    ROIstart = 0;               % Whole signal starts at 0
    ROIlength = t(end);         % ROI length is full duration
else
    startIndex = round(ROIstart * fs) + 1;
    endIndex   = round((ROIstart + ROIlength) * fs);
    xROI = audioData(startIndex:endIndex);
    tROI = t(startIndex:endIndex);
end

audioSegment = xROI;
% 
% % Define time axis for raw ROI signal
% tROI = linspace(startTime, endTime, length(audioSegment));
% xROI = audioSegment;

% Import provided labels
labels = importLabels(labelPath, fs);

%% ==================== BANDPASS FILTER ====================
bpFilter = designfilt('bandpassiir', 'FilterOrder', 12, ...
    'HalfPowerFrequency1', lowFreq, 'HalfPowerFrequency2', highFreq, ...
    'SampleRate', fs);
filteredAudioSegment = filtfilt(bpFilter, audioSegment);

%% ==================== BSCD CALCULATION ====================
tic
fprintf("Calculating BSCD...\n");
bscdOut = bscd(filteredAudioSegment.^2, wlen*fs);
toc


bscdOutMovMean = smoothdata(bscdOut, 'movmean', maWindow);

% Create time axis for BSCD output
startTime = ROIstart;
endTime = ROIstart + ROIlength;
t_bscd = linspace(startTime, endTime, length(bscdOutMovMean));
powerEnvelope = bscdOutMovMean;

%% ==================== DETECTION USING BSCD ====================
% Define a simple adaptive threshold and binary detections
optimalThreshold = mean(powerEnvelope) * ones(size(powerEnvelope));
binaryDetections = powerEnvelope > optimalThreshold;

% Simple event detection from binary detections
detectionDiff = diff([0, binaryDetections, 0]);
startIndices = find(detectionDiff == 1);
endIndices = find(detectionDiff == -1) - 1;
detectedLabels = struct('StartTime', cell(1, length(startIndices)), 'EndTime', cell(1, length(startIndices)));
for i = 1:length(startIndices)
    detectedLabels(i).StartTime = t_bscd(startIndices(i));
    detectedLabels(i).EndTime = t_bscd(endIndices(i));
end

%% ==================== POST-PROCESSING ====================
% Remove events shorter than minDuration and with very low effective power.
% validEvents = [];
% for i = 1:length(detectedLabels)
%     duration = detectedLabels(i).EndTime - detectedLabels(i).StartTime;
%     idx = find(t_bscd >= detectedLabels(i).StartTime & t_bscd <= detectedLabels(i).EndTime);
%     avgPower = mean(powerEnvelope(idx));
%     if duration >= minDuration && avgPower >= minEffectivePower
%         validEvents = [validEvents; detectedLabels(i)]; %#ok<AGROW>
%     end
% end
% detectedLabels = validEvents;

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

%% ==================== LOAD PROVIDED LABELS (GROUND TRUTH) AND FILTER TO ROI ====================
providedLabelsFull = importLabels(labelPath, fs);
if runWholeSignal
    providedLabelsROI = providedLabelsFull;
else
    ROIend = ROIstart + ROIlength;
    providedLabelsROI = providedLabelsFull(arrayfun(@(x) (x.StartTime >= ROIstart && x.EndTime <= ROIend), providedLabelsFull));
end
tempProvidedFile = fullfile(tempdir, "provided_labels.txt");
exportLabels(providedLabelsROI, tempProvidedFile);

%% ==================== STATISTICAL ANALYSIS ====================
tempDetectedFile = fullfile(tempdir, "detected_labels.txt");
tempProvidedFile = fullfile(tempdir, "provided_labels.txt");
exportLabels(detectedLabels, tempDetectedFile);
stats = compareLabels(tempProvidedFile, tempDetectedFile, fs);

%% ==================== PLOTTING (if enabled) ====================
if plotDetection
    % Plot BSCD results in one figure (4-panel layout)
    figure;
    tiledlayout(4,1);
    
    % Plot raw ROI signal
    nexttile;
    plot(tROI, xROI);
    if runWholeSignal
        title("Full Signal");
    else
        title("ROI Signal");
    end
    xlabel("Time (s)"); ylabel("Amplitude");
    grid on;
    
    % Plot smoothed BSCD power envelope & adaptive threshold
    nexttile;
    plot(t_bscd, powerEnvelope, 'LineWidth', 1.5);
    hold on;
    plot(t_bscd, optimalThreshold, 'k--', 'LineWidth', 1.5);
    hold off;
    title("BSCD Power Envelope & Adaptive Threshold");
    xlabel("Time (s)"); ylabel("Normalized Power");
    grid on;
    
    % Plot binary detection results
    nexttile;
    stem(t_bscd, binaryDetections, 'r');
    title("Binary Detections");
    xlabel("Time (s)"); ylabel("Detection (1/0)");
    grid on;
    
    % Plot detected events over raw ROI signal
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
    
    % ---------------------- Additional Spectrogram Plotting ----------------------
    % Define spectrogram parameters
    windowSize = 256;
    overlap = windowSize / 2;
    
    % Compute spectrogram of the original audio segment
    [s, f, t, p] = spectrogram(audioSegment, windowSize, overlap, [], fs, 'yaxis');
    f_kHz = f / 1000; % Convert frequency to kHz for plotting
    
    % Create a new figure for spectrogram and overlay
    figure('units','normalized','outerposition',[0 0 1 1])
    tiledlayout(3,1);
    
    % Plot raw audio segment
    nexttile;
    plot(audioSegment);
    axis tight;
    title('Audio Segment');
    xlabel('Samples');
    ylabel('Amplitude');
    
    % Plot spectrogram
    nexttile;
    imagesc(t, f_kHz, 10*log10(p));
    axis xy;
    ylim([40 120]); % Limit y-axis to the ultrasonic range
    colormap hsv;
    colorbar off;
    xlabel('Time (s)');
    ylabel('Frequency (kHz)');
    title('Spectrogram with Labels Overlayed');
    
    % Overlay Labels on Spectrogram
    hold on;
    for i = 1:length(labels)
        % Only plot labels within the specified time segment
        if labels(i).StartTime >= startTime && labels(i).EndTime <= endTime
            labelStart = labels(i).StartTime - startTime;
            labelEnd = labels(i).EndTime - startTime;
            % Plot vertical lines at label start and end times
            line([labelStart, labelStart], [0, 120], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
            line([labelEnd, labelEnd], [0, 120], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
            % Display label text centered between the lines
            text((labelStart+labelEnd)/2, 100, labels(i).Label, 'Color', 'red', 'FontSize', 8, ...
                 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
        end
    end
    hold off;
    
    % Plot BSCD output in the third tile of the spectrogram figure
    nexttile;
    plot(bscdOut);
    titleString = "BSCD, wlen = " + string(wlen*1000) + " ms";
    title(titleString);
end

%% ==================== OUTPUT ====================
exportLabels(detectedLabels, outputPath);