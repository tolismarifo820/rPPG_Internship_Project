%% rPPG Time and Frequency Domain Analysis (4-Level Decomposition)
% Reconstructs signals using cD3 and cD4 from a level 4 decomposition.
% Generates interactive subplots for time-series and frequency spectra.

clear; close all; clc;

%% ---------------- Configuration ----------------
% Target the root logs folder to process all participants automatically
base_path = 'C:\Users\tolis\Desktop\Praktiki\rPPG_project\logs'; 

waveletName = 'rbio6.8';
decompositionLevel = 4;
SIGNAL_COLUMN = 21; % active_signal
FPS_COLUMN = 16;    % actual_fps for frequency mapping

%% ---------------- Get all CSV files ----------------
files = dir(fullfile(base_path, '**\metadata.csv'));
numFiles = length(files);
fprintf('Found %d files to process.\n', numFiles);

%% ---------------- Sequential Reconstruction & Plotting ----------------
for fi = 1:numFiles
    try
        fname = files(fi).name;
        folder = files(fi).folder;
        fpath = fullfile(folder, fname);
        
        data = readtable(fpath);
        
        if isempty(data) || size(data, 2) < SIGNAL_COLUMN
            continue; 
        end
        
        % Extract signals and sampling rate
        sig = table2array(data(:, SIGNAL_COLUMN));
        sig(isnan(sig)) = 0;
        
        fps_array = table2array(data(:, FPS_COLUMN));
        Fs = mean(fps_array, 'omitnan');
        if isnan(Fs) || Fs == 0, Fs = 30; end % Fallback if FPS is missing
        
        % --- Wavelet decomposition (4 Levels) ---
        [C, L] = wavedec(sig, decompositionLevel, waveletName);
        
        d4_rec = wrcoef('d', C, L, waveletName, 4);
        d3_rec = wrcoef('d', C, L, waveletName, 3);
        
        if length(d4_rec) ~= length(sig)
            d4_rec = resize_signal(d4_rec, length(sig));
            d3_rec = resize_signal(d3_rec, length(sig));
        end
        
        recon_sig = d4_rec + d3_rec;
        
        % --- Frequency Spectrum Calculation (FFT) ---
        L_sig = length(sig);
        f = Fs * (0:(floor(L_sig/2))) / L_sig; % Frequency vector
        
        % Subtract mean to remove the DC offset (0 Hz spike)
        fft_orig = abs(fft(sig - mean(sig)) / L_sig);
        P1_orig = fft_orig(1:floor(L_sig/2)+1);
        P1_orig(2:end-1) = 2 * P1_orig(2:end-1);
        
        fft_recon = abs(fft(recon_sig - mean(recon_sig)) / L_sig);
        P1_recon = fft_recon(1:floor(L_sig/2)+1);
        P1_recon(2:end-1) = 2 * P1_recon(2:end-1);
        
        [~, parent_folder] = fileparts(folder);
        
        % --- Interactive Subplots ---
        figure('Name', sprintf('rPPG Analysis - %s', parent_folder), 'Position', [100, 100, 900, 600]);
        
        % Top Panel: Time Domain
        subplot(2, 1, 1);
        plot(sig, 'Color', [0.7 0.7 0.7], 'DisplayName', 'Original active_signal');
        hold on;
        plot(recon_sig, 'r', 'LineWidth', 1.5, 'DisplayName', 'Reconstructed (cD3 + cD4)');
        title(sprintf('Time Domain (%s, 4 Levels) - %s', waveletName, parent_folder), 'Interpreter', 'none');
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
        xlim([0 5]); % Limit to 0-5 Hz (0 to 300 BPM) for relevant cardiac band
        legend('Location', 'best');
        grid on;
        
        fprintf('Reconstructed and Plotted: %s\n', parent_folder);
        
    catch ME
        warning('Error processing %s: %s', fname, ME.message);
    end
end

fprintf('All files processed.\n');

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