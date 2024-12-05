function [x, t] = vocoder(signal, fs, spectralCutFreqHigh, spectralCutFreqLow, ST)
    % VOCODER Apply a vocoder effect to an input signal
    %
    % Args:
    %   signal: Input audio signal
    %   fs: Sampling frequency of the signal
    %   preemphasisConstant: Pre-emphasis filter constant
    %   spectralCutFreqHigh: Upper frequency limit for spectral cutoff
    %   spectralCutFreqLow: Lower frequency limit for spectral cutoff
    %   ST: Number of semitones for pitch shifting
    %
    % Returns:
    %   x: Output vocoder-processed signal
    %   t: Time vector for the output signal

    N = length(signal);              
    window_length = round(0.03 * fs); % 30 ms window length
    window_shift  = round(window_length / 4); % 75% overlap

    % Semitone pitch-shifting factors
    N1 = window_shift;
    N2 = round(2^(ST / 12) * N1);
    R  = N2 / N1;

    % Apply pre-emphasis
    %signal = filter([1 -preemphasisConstant], 1, signal);

    % Analysis
    X = [];
    k = 1;
    for start = 0:window_shift:N - window_length
        frame = signal(start + 1:start + window_length) .* hamming(window_length);
        X(:, k) = fft(frame);     
        k = k + 1;
    end

    % Transformation
    Xmag = interp1q([0:size(X, 2) - 1]', abs(X'), [0:1 / R:size(X, 2) - 2]');
    new_grid = floor(0:1 / R:size(X, 2) - 2) + 1;
    D = diff(angle(X'))';
    D_new = D(:, new_grid);
    phaseX = cumsum(D_new');
    Y = (Xmag .* exp(1j * phaseX))';

    % Spectral Cutoff
    spectralCutColumnHigh = round((fs - spectralCutFreqHigh) / (fs / window_length) + 1);
    spectralCutColumnLow  = round((fs - spectralCutFreqLow) / (fs / window_length) + 1);
    Y(1:spectralCutColumnHigh, :) = 0;
    Y(spectralCutColumnLow:end, :) = 0;

    % Synthesis
    x = zeros(N + window_length, 1);
    k = 1;
    for start = 0:window_shift:(size(Y, 2) - 1) * window_shift
        segment = real(ifft(Y(:, k), window_length)) .* hamming(window_length);
        x(start + 1:start + window_length) = x(start + 1:start + window_length) + segment;
        k = k + 1;
    end

    % Resample
    x = resample(x, N1, N2);
    x = x / max(abs(x)); % Normalize
    t = (0:length(x) - 1) / fs;
end