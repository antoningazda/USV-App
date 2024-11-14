function result = calculateMetrics(referenceLabels, detectedLabels)
    % Initialize Counters
    truePositives = 0;
    falsePositives = 0;
    falseNegatives = 0;

    % Prepare table to store detailed label information
    labelInfo = table('Size', [length(referenceLabels), 3], ...
                      'VariableTypes', {'double', 'logical', 'string'}, ...
                      'VariableNames', {'StartTime', 'Detected', 'DetectedLabel'});

    % Iterate through Reference Labels and Evaluate Detections
    for i = 1:length(referenceLabels)
        refStart = referenceLabels(i).StartTime;
        refEnd = referenceLabels(i).EndTime;
        refMidpoint = (refStart + refEnd) / 2;

        % Store the reference label's start time
        labelInfo.StartTime(i) = refStart;

        % Check if the midpoint of the reference label is inside any detected label
        detected = false;
        detectedLabel = "";

        for j = 1:length(detectedLabels)
            detStart = detectedLabels(j).StartTime;
            detEnd = detectedLabels(j).EndTime;

            % Midpoint criterion: Check if midpoint of reference label is within the detected label
            if refMidpoint >= detStart && refMidpoint <= detEnd
                detected = true;
                detectedLabel = detectedLabels(j).Label;
                break; % No need to check further once matched
            end
        end

        % Update detection status in the table
        labelInfo.Detected(i) = detected;
        labelInfo.DetectedLabel(i) = detectedLabel;

        % Update counts based on matches
        if detected
            truePositives = truePositives + 1;
        else
            falseNegatives = falseNegatives + 1;
        end
    end

    % Calculate False Positives
    % Detected labels that don’t match any reference label by midpoint criterion are false positives
    for j = 1:length(detectedLabels)
        detStart = detectedLabels(j).StartTime;
        detEnd = detectedLabels(j).EndTime;

        midpointMatches = any(arrayfun(@(x) (x.StartTime + x.EndTime) / 2 >= detStart && ...
                                              (x.StartTime + x.EndTime) / 2 <= detEnd, referenceLabels));
        if ~midpointMatches
            falsePositives = falsePositives + 1;
        end
    end

    % Calculate Metrics
    precision = truePositives / (truePositives + falsePositives);
    recall = truePositives / (truePositives + falseNegatives);
    if precision + recall > 0
        f1_score = 2 * (precision * recall) / (precision + recall);
    else
        f1_score = 0; % Handle case where precision and recall are zero
    end

    % Store Results in a Struct
    result = struct();
    result.TruePositives = truePositives;
    result.FalsePositives = falsePositives;
    result.FalseNegatives = falseNegatives;
    result.Precision = precision;
    result.Recall = recall;
    result.F1Score = f1_score;
    result.NumReferenceLabels = length(referenceLabels);
    result.NumDetectedLabels = length(detectedLabels);
    result.LabelInfo = labelInfo; % Table summarizing detection results for each reference label
end
