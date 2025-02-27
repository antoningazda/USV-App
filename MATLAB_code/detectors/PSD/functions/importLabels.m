function labelsStruct = importLabels(labelPath, fs)
    % Function to import label data from a text file and store it in a struct array.
    %
    % Args:
    %   labelPath (string): The path to the label file to be imported.
    %   fs (double): The sampling rate of the audio file.
    %
    % Returns:
    %   labelsStruct (struct array): A struct array where each struct contains
    %   start time, end time, start frequency, end frequency, label, start index, and stop index.

    % Define import options
    opts = delimitedTextImportOptions("NumVariables", 3);

    % Specify range and delimiter
    opts.DataLines = [1, Inf];
    opts.Delimiter = ["\t", "\\"];

    % Specify column names and types
    opts.VariableNames = ["A", "B", "C"];
    opts.VariableTypes = ["string", "string", "string"];

    % Specify file level properties
    opts.ExtraColumnsRule = "ignore";
    opts.EmptyLineRule = "read";
    opts.ConsecutiveDelimitersRule = "join";

    % Import the data
    tbl = readmatrix(labelPath, opts);

    % Extract times, frequencies, and label names
    labelTimes = str2double(tbl(1:2:end, 1:2)); % Start and end times
    labelsFrequencies = str2double(tbl(2:2:end, 2:3)); % Frequencies
    labelsNames = (tbl(1:2:end, 3)); % Labels (names)

    % Initialize struct array
    numLabels = size(labelTimes, 1);
    labelsStruct(numLabels) = struct('StartTime', [], 'EndTime', [], 'Label', '', ...
        'StartFrequency', [], 'EndFrequency', [], 'StartIndex', [], 'StopIndex', []); % Preallocation

    % Populate the struct array
    for i = 1:numLabels
        % Calculate start and stop indices based on sampling rate
        startIndex = round(labelTimes(i, 1) * fs); % Convert start time to sample index
        stopIndex = round(labelTimes(i, 2) * fs);  % Convert end time to sample index
        
        % Fill the struct
        labelsStruct(i).StartTime = labelTimes(i, 1); % Start time
        labelsStruct(i).EndTime = labelTimes(i, 2);   % End time
        labelsStruct(i).StartIndex = startIndex;       % Start index in samples
        labelsStruct(i).StopIndex = stopIndex;         % Stop index in samples
        labelsStruct(i).Label = labelsNames{i};       % Label name
        labelsStruct(i).StartFrequency = labelsFrequencies(i, 1); % Start frequency
        labelsStruct(i).EndFrequency = labelsFrequencies(i, 2);   % End frequency
        
    end

    % Clear temporary variables
    clear opts tbl labelTimes labelsFrequencies labelsNames
end
