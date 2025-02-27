function USVGUI(labels, audioData, fs)
    % Create a GUI for navigating and plotting USV segments with arrow keys
    %
    % Args:
    %   labels: Struct array containing the labels of each segment
    %   audioData: Audio data array
    %   fs: Sampling rate of the audio data
    
    % Initialize the segment index
    segmentIndex = 1;

    % Create the figure window
    hFig = figure('Name', 'USV Vocalization Segments', ...
                  'KeyPressFcn', @(src, event) keyPressCallback(src, event), ...
                  'NumberTitle', 'off', ...
                  'Position', [100, 100, 800, 600]);

    % Plot the initial segment
    plotSegment(labels, audioData, fs, segmentIndex);

    % Callback function to handle key press events
    function keyPressCallback(~, event)
        switch event.Key
            case 'rightarrow' % Right arrow key
                % Increment the segment index
                segmentIndex = segmentIndex + 1;
                if segmentIndex > length(labels)
                    segmentIndex = length(labels); % Prevent exceeding max index
                end
                % Replot the new segment
                clf; % Clear the figure
                plotSegment(labels, audioData, fs, segmentIndex);
            case 'leftarrow' % Left arrow key
                % Decrement the segment index
                segmentIndex = segmentIndex - 1;
                if segmentIndex < 1
                    segmentIndex = 1; % Prevent going below first index
                end
                % Replot the new segment
                clf; % Clear the figure
                plotSegment(labels, audioData, fs, segmentIndex);
        end
    end
end
