%
% % NOTES
% % Last Update: 29.11. 2024 1:37



fs = 250000; % Replace with the actual sampling rate of your audio

% Load Provided and Detected Labels using importLabels

path = "/Users/gazda/Documents/CTU/Masters/Masters thesis/";
datapath = path + "data/";
audioFile = "LPS-SI2homo-mH02-I04-USV";
labelFile = audioFile + "-IVojt";
labelPath = datapath + "labels/" + labelFile + ".txt";




providedLabels = importLabels(labelPath, fs);
detectedLabels = importLabels('output_C.txt', fs);

% Initialize Counters
truePositives_midpoint = 0;
truePositives_overlap = 0;
falsePositives = 0;
falseNegatives = 0;

% Iterate through Provided Labels and Evaluate Detections
for i = 1:length(providedLabels)
    provStart = providedLabels(i).StartTime;
    provEnd = providedLabels(i).EndTime;
    provMidpoint = (provStart + provEnd) / 2;
    provDuration = provEnd - provStart;

    % Check if the midpoint of the provided label is inside any detected label
    midpointMatched = false;
    overlapMatched = false;

    for j = 1:length(detectedLabels)
        detStart = detectedLabels(j).StartTime;
        detEnd = detectedLabels(j).EndTime;
        detDuration = detEnd - detStart;

        % Midpoint criterion: Check if midpoint of provided label is within the detected label
        if provMidpoint >= detStart && provMidpoint <= detEnd
            midpointMatched = true;
        end

        % 50% overlap criterion: Check if at least 50% overlap exists between provided and detected label
        overlap = max(0, min(provEnd, detEnd) - max(provStart, detStart));
        if overlap >= 0.5 * provDuration && overlap >= 0.5 * detDuration
            overlapMatched = true;
        end

        % If both criteria are met, break early
        if midpointMatched && overlapMatched
            break;
        end
    end

    % Update counts based on matches
    if midpointMatched
        truePositives_midpoint = truePositives_midpoint + 1;
    else
        falseNegatives = falseNegatives + 1;
    end

    if overlapMatched
        truePositives_overlap = truePositives_overlap + 1;
    end
end

% Calculate False Positives
% Detected labels that don’t match any provided label by midpoint criterion are false positives
for j = 1:length(detectedLabels)
    detStart = detectedLabels(j).StartTime;
    detEnd = detectedLabels(j).EndTime;

    midpointMatches = any(arrayfun(@(x) (x.StartTime + x.EndTime) / 2 >= detStart && ...
                                          (x.StartTime + x.EndTime) / 2 <= detEnd, providedLabels));
    if ~midpointMatches
        falsePositives = falsePositives + 1;
    end
end

% Calculate Precision, Recall, and F1-Score for Midpoint Criterion
precision_midpoint = truePositives_midpoint / (truePositives_midpoint + falsePositives);
recall_midpoint = truePositives_midpoint / (truePositives_midpoint + falseNegatives);
f1_score_midpoint = 2 * (precision_midpoint * recall_midpoint) / (precision_midpoint + recall_midpoint);

% Calculate Precision, Recall, and F1-Score for 50% Overlap Criterion
precision_overlap = truePositives_overlap / (truePositives_overlap + falsePositives);
recall_overlap = truePositives_overlap / (truePositives_overlap + falseNegatives);
f1_score_overlap = 2 * (precision_overlap * recall_overlap) / (precision_overlap + recall_overlap);

% Display Results
fprintf('Statistics based on Midpoint Criterion:\n');
fprintf('True Positives: %d\n', truePositives_midpoint);
fprintf('False Positives: %d\n', falsePositives);
fprintf('False Negatives: %d\n', falseNegatives);
fprintf('Precision: %.2f\n', precision_midpoint);
fprintf('Recall: %.2f\n', recall_midpoint);
fprintf('F1-Score: %.2f\n\n', f1_score_midpoint);

fprintf('Statistics based on 50%% Overlap Criterion:\n');
fprintf('True Positives: %d\n', truePositives_overlap);
fprintf('False Positives: %d\n', falsePositives);
fprintf('False Negatives: %d\n', falseNegatives);
fprintf('Precision: %.2f\n', precision_overlap);
fprintf('Recall: %.2f\n', recall_overlap);
fprintf('F1-Score: %.2f\n', f1_score_overlap);

