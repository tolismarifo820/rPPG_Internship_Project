%% rPPG Continuous Wavelet Transform (CWT) Multi-Wavelet Comparison (2x2 Grid)
% Linearly scaled frequency axis with automatic, uniform increments.

clear; close all; clc;

%% ---------------- Configuration ----------------
input_file = 'C:\Users\tolis\Desktop\Praktiki\rPPG_project\logs\P026\all_extracted_signals.csv';

video_duration = 30;  % Video duration in seconds
f_low = 0.1;          % Lower bound for CWT computation (must be > 0)
f_high = 3.0;         % Target upper frequency bound in Hz

%% ---------------- Load Data ----------------
if ~isfile(input_file)
    error('Input file not found: %s', input_file);
end

data = readtable(input_file);
headers = data.Properties.VariableNames;
numSignals = width(data);
numSamples = height(data);

% Dynamically calculate sampling rate from metadata
Fs = numSamples / video_duration;
fprintf('Loaded %d signal combinations.\n', numSignals);
fprintf('Detected %d samples over %d seconds. True Fs = %.2f FPS.\n', numSamples, video_duration, Fs);

if f_high >= (Fs / 2)
    f_high = (Fs / 2) - 0.1;
    fprintf('[WARN] f_high clamped to Nyquist limit: %.2f Hz.\n', f_high);
end

%% ---------------- Processing & Plotting Loop ----------------
for s = 1:numSignals
    combination_name = headers{s};
    sig = table2array(data(:, s));
    
    sig(isnan(sig)) = 0;
    sig = sig - mean(sig);
    N = length(sig);
    t = (0:N-1) / Fs;
    
    fig = figure('Name', sprintf('CWT 2x2 Comparison - %s', combination_name), ...
                 'Position', [100, 80, 1300, 850]);
    
    % --- Top Left (1): Filtered Temporal Signal ---
    subplot(2, 2, 1);
    plot(t, sig, 'k', 'LineWidth', 1.1);
    title(sprintf('Temporal Signal: %s', combination_name), 'Interpreter', 'none');
    xlabel('Time (s)');
    ylabel('Amplitude');
    xlim([0 max(t)]);
    grid on;
    
    % --- Subplots 2, 3, and 4 ---
    wavelets = {'amor', 'morse', 'bump'};
    titles = {'CWT: Analytic Morlet (amor)', 'CWT: Generalized Morse (morse)', 'CWT: Bump Wavelet (bump)'};
    sub_indices = [2, 3, 4];
    
    for w_idx = 1:3
        subplot(2, 2, sub_indices(w_idx));
        
        % Compute CWT coefficients, frequency vector, and COI
        [wt, f_cwt, coi] = cwt(sig, wavelets{w_idx}, Fs, 'FrequencyLimits', [f_low, f_high]);
        
        % pcolor maps non-uniform frequency coordinates into linear Cartesian space
        p = pcolor(t, f_cwt, abs(wt));
        set(p, 'EdgeColor', 'none');
        shading interp;
        
        % Linear frequency axis scaling with natural, uniform increments
        set(gca, 'YScale', 'linear');
        ylim([0 f_high]);
        yticks(0:1:f_high);
        
        hold on;
        % Overlay Cone of Influence
        plot(t, coi, '--w', 'LineWidth', 1.5);
        hold off;
        
        colormap(gca, 'parula');
        title(titles{w_idx});
        xlabel('Time (s)');
        ylabel('Frequency (Hz)');
        colorbar;
    end
    
    fprintf('Viewing [%d/%d]: %s\n', s, numSignals, combination_name);
    disp('Press SPACE or ENTER in the Command Window to view the next combination...');
    pause;
    
    close(fig);
end

fprintf('Finished analyzing all CWT combinations.\n');