function stats_midpoint = compareLabels(providedLabelPath, detectedLabelPath, fs)
    % Compare two label files and return statistics for detection accuracy.
    %
    % Args:
    %   providedLabelPath (string): Path to the provided label file.
    %   detectedLabelPath (string): Path to the detected label file.
    %   fs (double): Sampling rate of the audio file.
    %
    % Returns:
    %   stats_midpoint (struct): Statistics based on the midpoint criterion.
    %   stats_overlap (struct): Statistics based on the 50% overlap criterion.

    % Load Provided and Detected Labels using importLabels
    providedLabels = importLabels(providedLabelPath, fs);
    detectedLabels = importLabels(detectedLabelPath, fs);

    % Total number of labels
    totalProvidedLabels = length(providedLabels);
    totalDetectedLabels = length(detectedLabels);

    % Initialize Counters
    truePositives_midpoint = 0;
    truePositives_overlap = 0;
    falseNegatives = 0;
    falsePositives = 0;

    % Track matched detected labels to avoid double-counting
    matchedDetected_midpoint = false(1, totalDetectedLabels);
    matchedDetected_overlap = false(1, totalDetectedLabels);

    % Iterate through Provided Labels and Evaluate Detections
    for i = 1:totalProvidedLabels
        provStart = providedLabels(i).StartTime;
        provEnd = providedLabels(i).EndTime;
        provMidpoint = (provStart + provEnd) / 2;
        provDuration = provEnd - provStart;

        % Flags to track if the current provided label is matched
        midpointMatched = false;
        overlapMatched = false;

        for j = 1:totalDetectedLabels
            % Skip already matched detected labels
            if matchedDetected_midpoint(j) && matchedDetected_overlap(j)
                continue;
            end

            detStart = detectedLabels(j).StartTime;
            detEnd = detectedLabels(j).EndTime;
            detDuration = detEnd - detStart;

            % Midpoint criterion: Check if the midpoint of provided label is within the detected label
            if ~matchedDetected_midpoint(j) && provMidpoint >= detStart && provMidpoint <= detEnd
                midpointMatched = true;
                matchedDetected_midpoint(j) = true; % Mark detected label as matched for midpoint
                break;
            end

            % Calculate overlap
            overlap = max(0, min(provEnd, detEnd) - max(provStart, detStart));

            % Debugging: Print overlap values for inspection
            % fprintf('Provided label %d and detected label %d have an overlap of %.2f\n', i, j, overlap);

            % Check if the overlap meets 50% criterion for both labels
            if ~matchedDetected_overlap(j) && overlap >= 0.5 * provDuration && overlap >= 0.5 * detDuration
                overlapMatched = true;
                matchedDetected_overlap(j) = true; % Mark detected label as matched for overlap
                break; % Stop further comparisons once matched
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
    falsePositives = sum(~matchedDetected_midpoint);

    % Calculate Precision, Recall, and F1-Score for Midpoint Criterion
    precision_midpoint = truePositives_midpoint / (truePositives_midpoint + falsePositives);
    recall_midpoint = truePositives_midpoint / (truePositives_midpoint + falseNegatives);
    f1_score_midpoint = 2 * (precision_midpoint * recall_midpoint) / (precision_midpoint + recall_midpoint);

    % Calculate Precision, Recall, and F1-Score for 50% Overlap Criterion
    precision_overlap = truePositives_overlap / (truePositives_overlap + falsePositives);
    recall_overlap = truePositives_overlap / (truePositives_overlap + falseNegatives);
    f1_score_overlap = 2 * (precision_overlap * recall_overlap) / (precision_overlap + recall_overlap);

    % Organize Statistics into Output Structs
    stats_midpoint = struct('TotalProvidedLabels', totalProvidedLabels, ...
                            'TotalDetectedLabels', totalDetectedLabels, ...
                            'TruePositives', truePositives_midpoint, ...
                            'FalsePositives', falsePositives, ...
                            'FalseNegatives', falseNegatives, ...
                            'Precision', precision_midpoint, ...
                            'Recall', recall_midpoint, ...
                            'F1Score', f1_score_midpoint);

    stats_overlap = struct('TotalProvidedLabels', totalProvidedLabels, ...
                           'TotalDetectedLabels', totalDetectedLabels, ...
                           'TruePositives', truePositives_overlap, ...
                           'FalsePositives', falsePositives, ...
                           'FalseNegatives', falseNegatives, ...
                           'Precision', precision_overlap, ...
                           'Recall', recall_overlap, ...
                           'F1Score', f1_score_overlap);

    % Display Results
    fprintf('Statistics based on Midpoint Criterion:\n');
    disp(stats_midpoint);
    % 
    % fprintf('Statistics based on 50%% Overlap Criterion:\n');
    % disp(stats_overlap);

    % Discrepancy Check
    % Check for discrepancies in midpoint statistics
    if (stats_midpoint.TruePositives + stats_midpoint.FalseNegatives) ~= totalProvidedLabels
        fprintf('Discrepancy detected in midpoint criterion: True Positives + False Negatives does not equal Total Provided Labels.\n');
    end
    if (stats_midpoint.TruePositives + stats_midpoint.FalsePositives) ~= totalDetectedLabels
        fprintf('Discrepancy detected in midpoint criterion: True Positives + False Positives does not equal Total Detected Labels.\n');
    end

    % % Check for discrepancies in 50% overlap statistics
    % if (stats_overlap.TruePositives + stats_overlap.FalseNegatives) ~= totalProvidedLabels
    %     fprintf('Discrepancy detected in 50%% overlap criterion: True Positives + False Negatives does not equal Total Provided Labels.\n');
    % end
    % if (stats_overlap.TruePositives + stats_overlap.FalsePositives) ~= totalDetectedLabels
    %     fprintf('Discrepancy detected in 50%% overlap criterion: True Positives + False Positives does not equal Total Detected Labels.\n');
    % end
end
