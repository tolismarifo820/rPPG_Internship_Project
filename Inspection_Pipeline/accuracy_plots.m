base_path = 'C:\Users\tolis\Desktop\Praktiki\rPPG_project\experiment';

% --- CONFIGURATION ---
gt_hr_col = 'PR';       
gt_time_col = 'DATE';   
num_participants = 10;
window_size_sec = 30;   % 30-second window as established
min_hr_bpm = 45;        
max_hr_bpm = 210;       
% ---------------------

all_maes = []; 
all_mapes = [];
all_rmses = [];
all_pearsons = [];
signal_names = {};

% Cell array to store data for plotting later
plot_data = cell(num_participants, 1);

fprintf('Processing %d participants with FFT Heart Rate Extraction (30s window)...\n', num_participants);

for p = 1:num_participants
    p_str = sprintf('P%03d', p);
    
    p_dir = fullfile(base_path, p_str);
    gt_path = fullfile(p_dir, [p_str '_data.csv']);
    ext_path = fullfile(p_dir, [p_str '_extracted_signals.csv']);
    meta_path = fullfile(p_dir, [p_str '_metadata.json']);
    
    if ~isfile(gt_path) || ~isfile(ext_path) || ~isfile(meta_path)
        continue;
    end
    
    try
        meta = jsondecode(fileread(meta_path));
        start_time = datetime(meta.start_time, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
        end_time = start_time + seconds(meta.duration_sec);
        fps = meta.true_fps;
    catch
        continue;
    end
    
    opts = detectImportOptions(gt_path);
    opts.VariableNamingRule = 'preserve';
    gt_df = readtable(gt_path, opts);
    
    if ~isdatetime(gt_df.(gt_time_col))
        gt_df.(gt_time_col) = datetime(gt_df.(gt_time_col));
    end
    
    if iscell(gt_df.(gt_hr_col)) || ischar(gt_df.(gt_hr_col)) || isstring(gt_df.(gt_hr_col))
        gt_df.(gt_hr_col) = str2double(string(gt_df.(gt_hr_col)));
    end
    
    mask = (gt_df.(gt_time_col) >= start_time) & (gt_df.(gt_time_col) <= end_time);
    gt_synced = gt_df(mask, :);
    valid_gt_mask = ~isnan(gt_synced.(gt_hr_col)) & ~isnat(gt_synced.(gt_time_col));
    gt_synced = gt_synced(valid_gt_mask, :);
    
    if height(gt_synced) < 2
        continue;
    end
    
    gt_hr = gt_synced.(gt_hr_col);
    gt_times = gt_synced.(gt_time_col);
    
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
    
    pred_hr_matrix = nan(length(gt_times), length(signal_names));
    
    for i = 1:length(gt_times)
        t_center = gt_times(i);
        
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
    
    for s = 1:length(signal_names)
        pred_hr = pred_hr_matrix(:, s);
        valid_pair = ~isnan(pred_hr) & ~isnan(gt_hr);
        
        if sum(valid_pair) > 1 
            err = gt_hr(valid_pair) - pred_hr(valid_pair);
            abs_err = abs(err);
            
            all_maes(p, s) = mean(abs_err);
            all_mapes(p, s) = mean(abs_err ./ (gt_hr(valid_pair) + 1e-8)) * 100;
            all_rmses(p, s) = sqrt(mean(err.^2));
            
            R_mat = corrcoef(gt_hr(valid_pair), pred_hr(valid_pair));
            if numel(R_mat) == 4
                all_pearsons(p, s) = R_mat(1,2);
            end
        end
    end
    
    % --- STORE DATA FOR PLOTTING ---
    % Find the best signal for this specific participant to plot
    if any(~isnan(all_maes(p, :)))
        [~, p_best_idx] = min(all_maes(p, :));
        
        plot_data{p}.time = gt_times;
        plot_data{p}.gt_hr = gt_hr;
        plot_data{p}.pred_hr = pred_hr_matrix(:, p_best_idx);
        plot_data{p}.p_str = p_str;
        plot_data{p}.best_signal = signal_names{p_best_idx};
    end
end

% --- GENERATE SUBPLOTS ---
figure('Name', 'Ground Truth vs rPPG Extracted Heart Rate', 'Position', [100, 100, 1400, 800]);

for p = 1:num_participants
    if isempty(plot_data{p})
        continue;
    end
    
    % Create a 5x2 grid of subplots
    subplot(5, 2, p);
    
    % Plot Ground Truth (Black Line) and Extracted HR (Red Dashed)
    plot(plot_data{p}.time, plot_data{p}.gt_hr, 'k-', 'LineWidth', 1.5); 
    hold on;
    plot(plot_data{p}.time, plot_data{p}.pred_hr, 'r--', 'LineWidth', 1.5);
    
    % Formatting
    title(sprintf('%s (Best: %s)', plot_data{p}.p_str, plot_data{p}.best_signal), 'Interpreter', 'none', 'FontSize', 9);
    ylabel('HR (bpm)');
    grid on;
    
    % Add legend only to the first subplot to save space
    if p == 1
        legend('Ground Truth', 'rPPG (Extracted)', 'Location', 'best');
    end
    
    % Only add X-axis labels to the bottom row
    if p >= 9
        xlabel('Time');
    end
end

sgtitle('Heart Rate Tracking: Ground Truth vs. rPPG Algorithm (30s Window)', 'FontSize', 14, 'FontWeight', 'bold');