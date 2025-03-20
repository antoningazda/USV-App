function plotSegment(labels, audioData, fs, segmentIndex)
    % Function to plot the waveform, spectrogram, and frequency tracking of a labeled audio segment
    % with an additional subplot showing the output of the RBD (Recursive Bayesian Detector) filter.
    %
    % Args:
    %   labels (struct array): Struct array containing label information.
    %   audioData (array): Audio data array.
    %   fs (double): Sampling rate of the audio file.
    %   segmentIndex (int): Index of the label to plot.

    % Extract the start and stop indices for the given segment
    startIndex = labels(segmentIndex).StartIndex;
    stopIndex = labels(segmentIndex).StopIndex;

    % Extract the audio segment corresponding to the label
    audioSegment = audioData(startIndex:stopIndex);

    % Create a time vector for the segment (in seconds)
    segmentTime = (startIndex:stopIndex) / fs;

    % Plot the waveform of the audio segment
    subplot(5,1,1); % Create a subplot for the waveform
    plot(segmentTime, audioSegment);
    xlabel('Time (s)');
    ylabel('Amplitude');
    title(['USV Vocalisation (', num2str(segmentIndex), ') | label: ', labels(segmentIndex).Label, ...
           ' | Start Time: ', num2str(labels(segmentIndex).StartTime), 's | End Time: ', ...
           num2str(labels(segmentIndex).EndTime), 's']);
    grid on; axis tight;

    % Compute the initial spectrogram (without denoising)
    [s, f, t, p] = spectrogram(audioSegment, 256, [], [], fs, 'yaxis');
    f_kHz = f / 1000; % Convert frequency to kHz for plotting

    % Plot the plain spectrogram without any lines or processing
    subplot(5,1,2);
    imagesc(t, f_kHz, 10*log10(p));
    axis xy;
    ylim([40 120]); % Set y-axis limits in kHz
    title('Plain Spectrogram');
    xlabel('Time (s)');
    ylabel('Frequency (kHz)');
    colormap hsv;
    colorbar off;

    % Middle Spectrogram - Processed with Unsmooth Tracking Line
    subplot(5,1,3);
    % Limit the frequency range to above 40 kHz for maximum frequency selection
    validFreqIndices = f >= 40000; % Indices of frequencies above 40 kHz
    f_above_40kHz = f_kHz(validFreqIndices); % Frequency values above 40 kHz in kHz
    p_above_40kHz = p(validFreqIndices, :); % Power spectrum values only for frequencies above 40 kHz

    % Plot the spectrogram with custom frequency limits
    imagesc(t, f_kHz, 10*log10(p));
    axis xy; % Correct orientation
    ylim([40 120]); % Set y-axis limits in kHz
    title('Spectrogram with Unsmooth Tracking Line');
    xlabel('Time (s)');
    ylabel('Frequency (kHz)');
    colormap hsv;
    colorbar off;

    % Calculate MinPeakHeight with a fallback threshold
    calculatedThreshold_dB = mean(10*log10(p_above_40kHz(:))) + std(10*log10(p_above_40kHz(:)));
    defaultThreshold_dB = -50; % Default threshold if calculated one fails
    powerThreshold_dB = calculatedThreshold_dB; % Start with calculated threshold

    % Improved frequency tracking by detecting peaks above threshold
    maxFreq = nan(1, length(t));
    for i = 1:length(t)
        % Extract the power values for the current time bin above 40 kHz
        currentPower = 10*log10(p_above_40kHz(:, i));
        
        % Find peaks above the power threshold
        [peaks, locs] = findpeaks(currentPower, 'MinPeakHeight', powerThreshold_dB);
        
        % If no peaks found with calculated threshold, try the default threshold
        if isempty(peaks)
            [peaks, locs] = findpeaks(currentPower, 'MinPeakHeight', defaultThreshold_dB);
        end
        
        if ~isempty(peaks)
            % Select the most prominent peak's frequency
            [~, maxPeakIdx] = max(peaks);
            maxFreq(i) = f_above_40kHz(locs(maxPeakIdx));
        end
    end

    % Overlay the unsmoothed maximum frequency line on the middle spectrogram
    hold on;
    plot(t, maxFreq, 'r--', 'LineWidth', 1.5); % Red dashed line for better visibility
    hold off;

    % Ensure visibility of the line by setting axis limits to match the spectrogram
    xlim([t(1), t(end)]); % Ensure x-axis (time) aligns with the spectrogram

    % Lower Spectrogram - Processed with Smoothed Tracking Line
    subplot(5,1,4);
    
    % Plot the same spectrogram data
    imagesc(t, f_kHz, 10*log10(p));
    axis xy; % Correct orientation
    ylim([40 120]); % Set y-axis limits in kHz
    title('Spectrogram with Smoothed Tracking Line');
    xlabel('Time (s)');
    ylabel('Frequency (kHz)');
    colormap hsv;
    colorbar off;

    % Apply adaptive smoothing to the maximum frequency array
    smoothMaxFreq = smoothdata(maxFreq, 'movmean', 'omitnan'); % Adaptive smoothing with NaN handling

    % Apply an additional threshold for visualizing only high-power regions
    visibilityThreshold_dB = powerThreshold_dB; % Adjust for stricter visibility control
    visibleIndices = ~isnan(smoothMaxFreq) & (10*log10(max(p_above_40kHz)) > visibilityThreshold_dB);
    smoothMaxFreq(~visibleIndices) = NaN; % Set low-power regions to NaN

    % Overlay the smoothed maximum frequency line on the lower spectrogram
    hold on;
    plot(t, smoothMaxFreq, 'r-', 'LineWidth', 1.5); % Solid red line for smooth max frequency
    hold off;

    % Ensure visibility of the line by setting axis limits to match the spectrogram
    xlim([t(1), t(end)]); % Ensure x-axis (time) aligns with the spectrogram

    % Additional subplot for RBD output
    % Check if rbd function exists
    if exist('RBD', 'file') == 2
        % Apply the RBD function to detect change points in the signal
        window_length = 400;
        AR_order_left = 4;
        AR_order_right = 2;
        Bayesian_Evidence_order = 6;
        rbd_output = RBD(audioSegment, window_length, AR_order_left, AR_order_right, Bayesian_Evidence_order);

        % Plot the RBD output in a new subplot
        subplot(5,1,5);
        plot(segmentTime, rbd_output, 'b-');
        xlabel('Time (s)');
        ylabel('RBD Output');
        title('RBD (Recursive Bayesian Detector) Output');
        grid on;
        axis tight
    else
        % Display warning if rbd function is not found
        warning('RBD function not found. Skipping RBD output subplot.');
    end
end
