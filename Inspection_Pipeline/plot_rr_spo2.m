%% Slow Vitals Interactive Analysis
% Computes true framerate and interactively plots RR and SpO2 pairs per ROI.

clear; close all; clc;

%% ---------------- Configuration ----------------
% Set the file path manually for one participant at a time
input_file = 'C:\Users\tolis\Desktop\Praktiki\rPPG_project\experiment\P005\all_slow_vitals1.csv';

video_duration = 30;  % Total length of the video in seconds

%% ---------------- Load Data ----------------
if ~isfile(input_file)
    error('Input file not found: %s', input_file);
end

data = readtable(input_file);
headers = data.Properties.VariableNames;
numSamples = height(data);
 
% Dynamically calculate true sampling rate
Fs = numSamples / video_duration;
fprintf('Detected %d samples over %d seconds. True Fs = %.2f FPS.\n', numSamples, video_duration, Fs);

% Extract unique ROIs by finding columns that end with _RR
rois = {};
for i = 1:length(headers)
    if endsWith(headers{i}, '_RR')
        rois{end+1} = strrep(headers{i}, '_RR', '');
    end
end
numROIs = length(rois);
fprintf('Found %d ROI combinations.\n', numROIs);

%% ---------------- Processing & Plotting Loop ----------------
% Time vector for the signals using true Fs
t = (0:numSamples-1) / Fs;

for s = 1:numROIs
    current_roi = rois{s};
    
    rr_col = [current_roi, '_RR'];
    spo2_col = [current_roi, '_SpO2'];
    
    % Ensure both columns exist before attempting to plot
    if ~ismember(rr_col, headers) || ~ismember(spo2_col, headers)
        continue;
    end
    
    rr_sig = data.(rr_col);
    spo2_sig = data.(spo2_col);
    
    % --- Interactive Plotting ---
    fig = figure('Name', sprintf('Slow Vitals - %s', current_roi), 'Position', [100, 100, 1000, 600]);
    
    % Top Subplot: Respiration Rate (RR)
    subplot(2, 1, 1);
    plot(t, rr_sig, 'b', 'LineWidth', 1.5);
    title(sprintf('Respiration Rate - %s', current_roi), 'Interpreter', 'none');
    xlabel('Time (Seconds)');
    ylabel('RR (Br/min)');
    xlim([0 max(t)]);
    ylim([5 45]);
    grid on;
    
    % Bottom Subplot: SpO2
    subplot(2, 1, 2);
    plot(t, spo2_sig, 'r', 'LineWidth', 1.5);
    title(sprintf('SpO2 - %s', current_roi), 'Interpreter', 'none');
    xlabel('Time (Seconds)');
    ylabel('SpO2 (%)');
    xlim([0 max(t)]);
    ylim([85 100]);
    grid on;
    
    fprintf('Viewing [%d/%d]: %s\n', s, numROIs, current_roi);
    disp('Press SPACE or ENTER in the Command Window to view the next combination...');
    pause;
    
    close(fig);
end

fprintf('Finished analyzing all ROIs.\n');