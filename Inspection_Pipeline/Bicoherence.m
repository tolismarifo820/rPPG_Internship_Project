%% rPPG High-Resolution Bicoherence Analysis
% Computes and plots a high-resolution 2D Bicoherence matrix for all extracted signal combinations.
% Uses NFFT zero-padding and interpolated pseudocolor shading without saving files to disk.

clear; close all; clc;

%% ---------------- Configuration ----------------
input_file = 'C:\Users\tolis\Desktop\Praktiki\rPPG_project\logs\P026\all_extracted_signals.csv';

video_duration = 30;  % Total length of the video in seconds
nfft = 1024;          % High-resolution frequency grid via zero-padding
f_low = 0.7;          % Lower bound for cardiac band (~42 BPM)
f_high = 2.5;         % Upper bound for cardiac band (~150 BPM)

%% ---------------- Load Data ----------------
if ~isfile(input_file)
    error('Input file not found: %s', input_file);
end

data = readtable(input_file);
headers = data.Properties.VariableNames;
numSignals = width(data);
numSamples = height(data);

% Dynamically calculate true sampling rate (fixes the 15 vs 30 FPS doubling issue)
Fs = numSamples / video_duration; 
fprintf('Loaded %d signal combinations.\n', numSignals);
fprintf('Detected %d samples over %d seconds. True Fs = %.2f FPS.\n', numSamples, video_duration, Fs);

%% ---------------- Processing & Plotting Loop ----------------
for s = 1:numSignals
    combination_name = headers{s};
    sig = table2array(data(:, s));
    
    % Replace any stray NaNs with 0 to prevent FFT crashes
    sig(isnan(sig)) = 0; 
    
    % 1. Windowing & Preprocessing
    sig = sig - mean(sig);
    N = length(sig);
    
    % Dynamically scale the Welch window to exactly 10 seconds based on true Fs
    win_len = min(round(Fs * 10), N); 
    noverlap = round(win_len * 0.75);
    hop = win_len - noverlap;

    num_segs = floor((N - win_len) / hop) + 1;
    
    % Frequency axis determined by NFFT resolution
    freqs = (0:(nfft/2)) * (Fs / nfft);

    % Isolate physiological cardiac band
    f_cardiac_idx = find(freqs >= f_low & freqs <= f_high);
    num_f = length(f_cardiac_idx);

    if num_f == 0 || num_segs < 1
        fprintf('Signal %s is too short or invalid.\n', combination_name);
        continue;
    end

    B = zeros(num_f, num_f); 
    P = zeros(length(freqs), 1); 

    % 2. Compute Segmented Bispectrum with NFFT Zero-Padding
    for k = 1:num_segs
        idx_start = (k - 1) * hop + 1;
        seg = sig(idx_start : idx_start + win_len - 1);
        seg = seg .* hann(win_len);
        
        X = fft(seg, nfft);
        X = X(1 : nfft/2 + 1);
        P = P + abs(X).^2;
        
        for i = 1:num_f
            idx_f1 = f_cardiac_idx(i);
            f1 = freqs(idx_f1);
            
            for j = 1:num_f
                idx_f2 = f_cardiac_idx(j);
                f2 = freqs(idx_f2);
                
                f3 = f1 + f2;
                if f3 <= (Fs / 2)
                    [~, idx_f3] = min(abs(freqs - f3));
                    B(i, j) = B(i, j) + (X(idx_f1) * X(idx_f2) * conj(X(idx_f3)));
                end
            end
        end
    end

    B = B / num_segs;
    P = P / num_segs;

    % 3. Calculate Kim & Powers Normalized Bicoherence (0 to 1)
    bicoh = zeros(num_f, num_f);
    for i = 1:num_f
        idx_f1 = f_cardiac_idx(i);
        for j = 1:num_f
            idx_f2 = f_cardiac_idx(j);
            [~, idx_f3] = min(abs(freqs - (freqs(idx_f1) + freqs(idx_f2))));
            denom = sqrt(P(idx_f1) * P(idx_f2) * P(idx_f3));
            if denom > 1e-8
                bicoh(i, j) = abs(B(i, j)) / denom;
            end
        end
    end

    % --- 4. Calculate Spectrogram (STFT) ---
    win_len_spec = round(Fs * 5);           % 5-second window dynamically scaled
    noverlap_spec = round(win_len_spec * 0.9); 
    nfft_spec = 2048;                       
    
    [S, F_spec, T_spec] = spectrogram(sig, hanning(win_len_spec), noverlap_spec, nfft_spec, Fs);
    F_bpm_spec = F_spec * 60; % Convert Hz to BPM
    
    % --- 5. Interactive Unified Plotting (2x2 Grid) ---
    fig = figure('Name', sprintf('Spectral Analysis - %s', combination_name), 'Position', [100, 100, 1400, 800]);
    
    % Time vector for the raw signal using true Fs
    t = (0:length(sig)-1) / Fs;
    
    % --- Top Left: Raw Signal ---
    subplot(2, 2, 1);
    plot(t, sig, 'k', 'LineWidth', 1.2);
    title(sprintf('Raw Signal - %s', combination_name), 'Interpreter', 'none');
    xlabel('Time (Seconds)');
    ylabel('Amplitude');
    xlim([0 max(t)]);
    grid on;
    
    % --- Top Right: Spectrogram ---
    subplot(2, 2, 2);
    imagesc(T_spec, F_bpm_spec, 10*log10(abs(S)));
    axis xy; 
    
    % Limit the Y-axis to physiological human cardiac frequencies (0 - 200 BPM)
    ylim([0 200]); 
    colormap(gca, 'jet'); 
    c1 = colorbar;
    ylabel(c1, 'Power (dB)');
    title('Spectrogram (STFT) Tracking');
    xlabel('Time (Seconds)');
    ylabel('Frequency (BPM)');
    
    % --- Bottom Left: Bispectrum Magnitude ---
    f_cardiac_bpm = freqs(f_cardiac_idx) * 60;
    
    subplot(2, 2, 3);
    pcolor(f_cardiac_bpm, f_cardiac_bpm, abs(B)); 
    shading interp;
    axis xy equal tight;
    colormap(gca, 'parula'); 
    c2 = colorbar;
    ylabel(c2, 'Bispectrum Magnitude');
    title('Bispectrum Magnitude');
    xlabel('f_1 (BPM)');
    ylabel('f_2 (BPM)');
    grid on;
    
    % --- Bottom Right: Normalized Bicoherence ---
    subplot(2, 2, 4);
    pcolor(f_cardiac_bpm, f_cardiac_bpm, bicoh); 
    shading interp;
    axis xy equal tight;
    colormap(gca, 'parula'); 
    c3 = colorbar;
    ylabel(c3, 'Squared Bicoherence (0-1)');
    caxis([0 1]); 
    title('Normalized Bicoherence');
    xlabel('f_1 (BPM)');
    ylabel('f_2 (BPM)');
    grid on;
    
    fprintf('Viewing [%d/%d]: %s\n', s, numSignals, combination_name);
    disp('Press SPACE or ENTER in the Command Window to view the next combination...');
    pause; 
    
    close(fig);
end

fprintf('Finished analyzing all combinations.\n'); 