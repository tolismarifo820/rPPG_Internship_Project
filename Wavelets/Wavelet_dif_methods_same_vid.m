%% rPPG Signal Analysis (4-Level Decomposition)
% Applies cD4 reconstruction to all 60 extracted signal combinations.
% Plots Time and Frequency domains interactively without saving the data.

clear; close all; clc;

%% ---------------- Configuration ----------------
input_file = 'C:\Users\tolis\Desktop\Praktiki\rPPG_project\logs\P031\all_extracted_signals.csv';

waveletName = 'dmey';
decompositionLevel = 4;
Fs = 30; % Locked to 30 FPS based on your camera hardware configuration

%% ---------------- Load Data ----------------
if ~isfile(input_file)
    error('Input file not found: %s', input_file);
end

data = readtable(input_file);
headers = data.Properties.VariableNames;
numSignals = width(data);

fprintf('Loaded %d signal combinations from P028.\n', numSignals);

%% ---------------- Processing & Plotting Loop ----------------
% We process them one by one. MATLAB will pause after each plot 
% to prevent 60 windows from opening simultaneously and crashing your system.
for i = 1:numSignals
    combination_name = headers{i};
    sig = table2array(data(:, i));
    sig(isnan(sig)) = 0; 
    
    % --- Wavelet decomposition (4 Levels) ---
    [C, L] = wavedec(sig, decompositionLevel, waveletName);
    
    % Reconstruct keeping ONLY detail level 4 (0.93 Hz to 1.87 Hz)
    d4_rec = wrcoef('d', C, L, waveletName, 4);
    
    if length(d4_rec) ~= length(sig)
        d4_rec = resize_signal(d4_rec, length(sig));
    end
    
    recon_sig = d4_rec;
    
    % --- Frequency Spectrum Calculation (FFT) ---
    L_sig = length(sig);
    f = Fs * (0:(floor(L_sig/2))) / L_sig; 
    
    fft_orig = abs(fft(sig - mean(sig)) / L_sig);
    P1_orig = fft_orig(1:floor(L_sig/2)+1);
    P1_orig(2:end-1) = 2 * P1_orig(2:end-1);
    
    fft_recon = abs(fft(recon_sig - mean(recon_sig)) / L_sig);
    P1_recon = fft_recon(1:floor(L_sig/2)+1);
    P1_recon(2:end-1) = 2 * P1_recon(2:end-1);
    
    % --- Interactive Subplots ---
    fig = figure('Name', sprintf('Analysis - %s', combination_name), 'Position', [100, 100, 1000, 600]);
    
    % Top Panel: Time Domain
    subplot(2, 1, 1);
    plot(sig, 'Color', [0.7 0.7 0.7], 'DisplayName', 'Original Signal');
    hold on;
    plot(recon_sig, 'r', 'LineWidth', 1.5, 'DisplayName', 'Reconstructed (cD4)');
    title(sprintf('Time Domain - %s', combination_name), 'Interpreter', 'none');
    xlabel('Frames');
    ylabel('Amplitude');
    legend('Location', 'best');
    grid on;
    
    % Bottom Panel: Frequency Domain
    subplot(2, 1, 2);
    plot(f, P1_orig, 'Color', [0.7 0.7 0.7], 'DisplayName', 'Original Spectrum');
    hold on;
    plot(f, P1_recon, 'r', 'LineWidth', 1.5, 'DisplayName', 'Reconstructed Spectrum');
    title('Frequency Domain (FFT)');
    xlabel('Frequency (Hz)');
    ylabel('Magnitude');
    xlim([0 5]); % Lock view to physiological band
    legend('Location', 'best');
    grid on;
    
    % Pause to allow viewing before generating the next one
    fprintf('Viewing [%d/%d]: %s\n', i, numSignals, combination_name);
    disp('Press SPACE or ENTER in the Command Window to view the next combination...');
    pause; 
    
    % Close the figure to keep your RAM clear
    close(fig);
end

fprintf('Finished analyzing all combinations.\n');

%% ---------------- Helper Function ----------------
function out = resize_signal(in, targetLen)
    if length(in) == targetLen
        out = in;
    elseif length(in) < targetLen
        out = [in; zeros(targetLen - length(in), 1)];
    else
        out = in(1:targetLen);
    end
end