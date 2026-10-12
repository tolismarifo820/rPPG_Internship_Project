%% Plot pulse oximeter data (30-Second Window)
% File path: C:\Users\tolis\Desktop\Praktiki\rPPG_project\experiment\P002\data.csv

clear;
clc;
close all;

%% File path

filename = "C:\Users\tolis\Desktop\Praktiki\rPPG_project\experiment\P006\data.csv";

if ~exist(filename, "file")
    error("File not found: %s", filename);
end

%% Detect import options and inspect variable names

opts = detectImportOptions(filename);

%% Read the table first (without special TreatAsMissing)

data = readtable(filename, opts);

%% Now set missing-value handling using the actual variable names

% Find which columns correspond to SPO2, PR, SI (case-insensitive)
varNames = opts.VariableNames;

idx_SPO2 = find(strcmpi(varNames, "SPO2"));
idx_PR   = find(strcmpi(varNames, "PR"));
idx_SI   = find(strcmpi(varNames, "SI"));

if isempty(idx_SPO2) || isempty(idx_PR) || isempty(idx_SI)
    error("Could not find columns SPO2, PR, SI. Detected names: %s", strjoin(varNames, ", "));
end

name_SPO2 = varNames{idx_SPO2};
name_PR   = varNames{idx_PR};
name_SI   = varNames{idx_SI};

% Re-detect options to modify them
opts = detectImportOptions(filename);

opts = setvaropts(opts, name_SPO2, "TreatAsMissing", "--");
opts = setvaropts(opts, name_PR,   "TreatAsMissing", "--");
opts = setvaropts(opts, name_SI,   "TreatAsMissing", "--");

% Ensure DATE is read as datetime (adjust name if needed)
idx_DATE = find(strcmpi(varNames, "DATE"));
if ~isempty(idx_DATE)
    name_DATE = varNames{idx_DATE};
    opts = setvaropts(opts, name_DATE, "InputFormat", "yyyy-MM-dd HH:mm:ss");
end

% Read again with proper options
data = readtable(filename, opts);

%% Convert columns to numeric (in case any are still text)

data.(name_SPO2) = convertToNumeric(data.(name_SPO2));
data.(name_PR)   = convertToNumeric(data.(name_PR));
data.(name_SI)   = convertToNumeric(data.(name_SI));

% Ensure DATE is datetime
if ~isdatetime(data.(name_DATE))
    data.(name_DATE) = datetime(data.(name_DATE), ...
        "InputFormat", "yyyy-MM-dd HH:mm:ss");
end

%% Relative time in seconds

time_sec = seconds(data.(name_DATE) - data.(name_DATE)(1));

%% User Input for 30-Second Window

fprintf('\nTotal recorded time: %.1f seconds.\n', time_sec(end));
start_time = input('Enter the starting time in seconds (e.g., 0): ');
end_time = start_time + 30;

% Find indices within the 30-second window
valid_idx = time_sec >= start_time & time_sec <= end_time;

if ~any(valid_idx)
    error('No data found in the specified time range (%.1f to %.1f seconds).', start_time, end_time);
end

% Filter the data
t_win = time_sec(valid_idx);
spo2_win = data.(name_SPO2)(valid_idx);
pr_win = data.(name_PR)(valid_idx);
si_win = data.(name_SI)(valid_idx);

%% Display basic info

fprintf("\nFile: %s\n", filename);
fprintf("Plotting window: %.1f s to %.1f s\n", start_time, end_time);
fprintf("Samples in window: %d\n", sum(valid_idx));

%% Plot signals vs elapsed time

figure("Color", "w", "Name", sprintf("Pulse Oximeter (%.1f s to %.1f s)", start_time, end_time));
tiledlayout(3, 1, "TileSpacing", "compact", "Padding", "compact");

% SpO2
nexttile;
plot(t_win, spo2_win, "b.-", "LineWidth", 1.2, "MarkerSize", 10);
grid on;
ylabel("SpO_2 (%)");
title("Oxygen Saturation");
xlim([start_time end_time]);
ylim padded;

% Pulse rate
nexttile;
plot(t_win, pr_win, "r.-", "LineWidth", 1.2, "MarkerSize", 10);
grid on;
ylabel("PR (bpm)");
title("Pulse Rate");
xlim([start_time end_time]);
ylim padded;

% Signal index
nexttile;
plot(t_win, si_win, "k.-", "LineWidth", 1.2, "MarkerSize", 10);
grid on;
ylabel("SI");
xlabel("Time from first sample (s)");
title("Signal Index");
xlim([start_time end_time]);
ylim padded;

%% Helper function

function numericData = convertToNumeric(inputData)
    if isnumeric(inputData)
        numericData = inputData;
        return;
    end
    numericData = str2double(string(inputData));
end