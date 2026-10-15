% Set the base directory for the experiment data
base_path = 'C:\Users\tolis\Desktop\Praktiki\rPPG_project\experiment';

% --- CONFIGURATION ---
gt_hr_col = 'PR';       % Heart Rate column from P0XX_data.csv
gt_time_col = 'DATE';   % Timestamp column from P0XX_data.csv
num_participants = 10;
window_size_sec = 30;   % 10-second window for FFT
min_hr_bpm = 45;        % Minimum human HR (0.75 Hz)
max_hr_bpm = 180;       % Maximum human HR (3 Hz)
% ---------------------

all_maes = []; 
all_mapes = [];
all_rmses = [];
all_pearsons = [];
signal_names = {};

fprintf('Processing %d participants with FFT Heart Rate Extraction...\n', num_participants);

for p = 1:num_participants
    p_str = sprintf('P%03d', p);
    
    p_dir = fullfile(base_path, p_str);
    gt_path = fullfile(p_dir, [p_str '_data.csv']);
    ext_path = fullfile(p_dir, [p_str '_extracted_signals.csv']);
    meta_path = fullfile(p_dir, [p_str '_metadata.json']);
    
    if ~isfile(gt_path) || ~isfile(ext_path) || ~isfile(meta_path)
        warning('Files missing for %s. Skipping.', p_str);
        continue;
    end
    
    % 1. Load and parse metadata
    try
        meta = jsondecode(fileread(meta_path));
        start_time = datetime(meta.start_time, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
        end_time = start_time + seconds(meta.duration_sec);
        fps = meta.true_fps;
    catch
        continue;
    end
    
    % 2. Load Ground Truth data (1 Hz)
    opts = detectImportOptions(gt_path);
    opts.VariableNamingRule = 'preserve';
    gt_df = readtable(gt_path, opts);
    
    if ~isdatetime(gt_df.(gt_time_col))
        gt_df.(gt_time_col) = datetime(gt_df.(gt_time_col));
    end
    
    if iscell(gt_df.(gt_hr_col)) || ischar(gt_df.(gt_hr_col)) || isstring(gt_df.(gt_hr_col))
        gt_df.(gt_hr_col) = str2double(string(gt_df.(gt_hr_col)));
    end
    
    % 3. Filter GT to window AND drop missing/NaN values
    mask = (gt_df.(gt_time_col) >= start_time) & (gt_df.(gt_time_col) <= end_time);
    gt_synced = gt_df(mask, :);
    valid_gt_mask = ~isnan(gt_synced.(gt_hr_col)) & ~isnat(gt_synced.(gt_time_col));
    gt_synced = gt_synced(valid_gt_mask, :);
    
    if height(gt_synced) < 2
        continue;
    end
    
    gt_hr = gt_synced.(gt_hr_col);
    gt_times = gt_synced.(gt_time_col);
    
    % 4. Load extracted signals (~30 Hz)
    opts_ext = detectImportOptions(ext_path);
    opts_ext.VariableNamingRule = 'preserve';
    ext_df = readtable(ext_path, opts_ext);
    
    ignore_cols = {'Time', 'Frame', 'Timestamp'};
    all_cols = ext_df.Properties.VariableNames;
    valid_cols = setdiff(all_cols, ignore_cols, 'stable');
    
    if isempty(signal_names)
        signal_names = valid_cols;
        num_sigs = length(signal_names);
        all_maes = nan(num_participants, num_sigs);
        all_mapes = nan(num_participants, num_sigs);
        all_rmses = nan(num_participants, num_sigs);
        all_pearsons = nan(num_participants, num_sigs);
    end
    
    ext_times = start_time + seconds((0:height(ext_df)-1) / fps)';
    ext_matrix = table2array(ext_df(:, valid_cols));
    
    % --- FFT HEART RATE EXTRACTION ---
    pred_hr_matrix = nan(length(gt_times), length(signal_names));
    
    for i = 1:length(gt_times)
        t_center = gt_times(i);
        
        % 10-second window centered on this GT reading
        window_mask = (ext_times >= (t_center - seconds(window_size_sec/2))) & ...
                      (ext_times <= (t_center + seconds(window_size_sec/2)));
        
        chunk = ext_matrix(window_mask, :);
        N = size(chunk, 1);
        
        if N < (fps * 3) 
            continue;
        end
        
        chunk = detrend(chunk);
        win = hamming(N);
        chunk = chunk .* win;
        
        Y = abs(fft(chunk, N, 1));
        frequencies = (0:N-1)' * (fps / N);
        
        valid_f_mask = (frequencies >= min_hr_bpm/60) & (frequencies <= max_hr_bpm/60);
        valid_freqs = frequencies(valid_f_mask);
        Y_valid = Y(valid_f_mask, :);
        
        if isempty(Y_valid)
            continue;
        end
        
        [~, max_idx] = max(Y_valid, [], 1);
        pred_hr_matrix(i, :) = valid_freqs(max_idx) * 60;
    end
    
    % 5. Compute Advanced Metrics
    for s = 1:length(signal_names)
        pred_hr = pred_hr_matrix(:, s);
        valid_pair = ~isnan(pred_hr) & ~isnan(gt_hr);
        
        if sum(valid_pair) > 1 % Need at least 2 points for correlation
            err = gt_hr(valid_pair) - pred_hr(valid_pair);
            abs_err = abs(err);
            
            all_maes(p, s) = mean(abs_err);
            all_mapes(p, s) = mean(abs_err ./ (gt_hr(valid_pair) + 1e-8)) * 100;
            all_rmses(p, s) = sqrt(mean(err.^2));
            
            % Pearson Correlation
            R_mat = corrcoef(gt_hr(valid_pair), pred_hr(valid_pair));
            if numel(R_mat) == 4
                all_pearsons(p, s) = R_mat(1,2);
            end
        end
    end
end

% --- DISPLAY RESULTS PER PARTICIPANT ---
fprintf('\n====================================================================================================\n');
fprintf('🏆 BEST PERFORMING SIGNAL PER PARTICIPANT 🏆\n');
fprintf('====================================================================================================\n');
fprintf('%-6s | %-32s | %-9s | %-9s | %-9s | %-10s\n', 'Subj', 'Best Signal (by MAE)', 'MAE (bpm)', 'MAPE (%)', 'RMSE', 'Pearson r');
fprintf('----------------------------------------------------------------------------------------------------\n');

for p = 1:num_participants
    p_str = sprintf('P%03d', p);
    
    if p > size(all_maes, 1) || all(isnan(all_maes(p, :)))
        fprintf('%-6s | %-32s | %-9s | %-9s | %-9s | %-10s\n', p_str, 'No Valid Data', '-', '-', '-', '-');
        continue;
    end
    
    % Rank by MAE
    [p_best_mae, p_best_idx] = min(all_maes(p, :));
    p_best_mape = all_mapes(p, p_best_idx);
    p_best_rmse = all_rmses(p, p_best_idx);
    p_best_pearson = all_pearsons(p, p_best_idx);
    p_best_signal = signal_names{p_best_idx};
    
    fprintf('%-6s | %-32s | %-9.2f | %-9.2f | %-9.2f | %-10.3f\n', ...
        p_str, p_best_signal, p_best_mae, p_best_mape, p_best_rmse, p_best_pearson);
end
fprintf('====================================================================================================\n\n');

% --- AGGREGATE RESULTS (OVERALL) ---
mean_maes = mean(all_maes, 1, 'omitnan');
mean_mapes = mean(all_mapes, 1, 'omitnan');
mean_rmses = mean(all_rmses, 1, 'omitnan');
mean_pearsons = mean(all_pearsons, 1, 'omitnan');

[overall_best_mae, overall_best_idx] = min(mean_maes);
overall_best_mape = mean_mapes(overall_best_idx);
overall_best_rmse = mean_rmses(overall_best_idx);
overall_best_pearson = mean_pearsons(overall_best_idx);
overall_best_signal = signal_names{overall_best_idx};

fprintf('OVERALL BEST SIGNAL (Averaged across all participants):\n');
fprintf('Signal Name: %s\n', overall_best_signal);
fprintf('Mean MAE:    %.2f bpm\n', overall_best_mae);
fprintf('Mean RMSE:   %.2f bpm\n', overall_best_rmse);
fprintf('Mean MAPE:   %.2f%%\n', overall_best_mape);
fprintf('Pearson r:   %.3f\n\n', overall_best_pearson);

fprintf('String for your LaTeX presentation (Overall Average):\n');
fprintf('%.1f\\%% MAPE / %.1f MAE / %.1f RMSE\n', overall_best_mape, overall_best_mae, overall_best_rmse);