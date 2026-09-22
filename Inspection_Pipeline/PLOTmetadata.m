% Base path where the metadata file is located (Update as needed)
basePath = 'C:\Users\tolis\Desktop\Praktiki\rPPG_project\logs\P025\';

% Target metadata file
fileName = 'metadata.csv';
fullPath = fullfile(basePath, fileName);

% Initialize cell arrays to hold signal data and their labels
allSignals = {};
allTitles = {};

% 1. Read the metadata file and extract every numeric column
if isfile(fullPath)
    % Read file, keeping original column headers
    dataTable = readtable(fullPath, 'PreserveVariableNames', true);
    
    % Keep only numeric columns
    numericData = dataTable(:, vartype('numeric'));
    
    for c = 1:width(numericData)
        % Store the data array
        allSignals{end+1} = numericData{:, c};
        
        % Create a title indicating the file and column name
        colName = strrep(numericData.Properties.VariableNames{c}, '_', '\_');
        fileNameClean = strrep(fileName, '_', '\_');
        allTitles{end+1} = ['[', fileNameClean, '] ', colName];
    end
else
    warning('File not found: %s', fullPath);
end

% 2. Plot the collected metadata columns in chunks of 3
numSignals = length(allSignals);

if numSignals > 0
    signalsPerFigure = 3;
    numFigures = ceil(numSignals / signalsPerFigure);
    
    for figIdx = 1:numFigures
        % Create a new figure with an offset position
        figure('Name', sprintf('Metadata Plots - Part %d', figIdx), ...
            'Color', 'w', 'Position', [100 + (figIdx*30), 100 + (figIdx*30), 900, 800]);
        
        % Calculate start and end indices for the current chunk
        startIdx = (figIdx - 1) * signalsPerFigure + 1;
        endIdx = min(figIdx * signalsPerFigure, numSignals);
        
        subplotIdx = 1;
        
        for s = startIdx:endIdx
            subplot(signalsPerFigure, 1, subplotIdx);
            
            % Plot with a line and marker ('.-') so single-point summaries are visible
            plot(allSignals{s}, '.-', 'LineWidth', 1.2, 'MarkerSize', 10);
            
            % Formatting
            title(allTitles{s}, 'FontWeight', 'bold');
            xlabel('Samples');
            ylabel('Value');
            grid on;
            
            % Adjust x-axis limits tightly, but handle single-point arrays safely
            if length(allSignals{s}) > 1
                xlim([1, length(allSignals{s})]);
            end
            
            subplotIdx = subplotIdx + 1;
        end
        
        % Main figure title
        sgtitle(sprintf('Metadata Signals (Plots %d to %d of %d)', ...
            startIdx, endIdx, numSignals), 'FontSize', 14, 'FontWeight', 'bold');
    end
else
    disp('No numeric signal data found in metadata.csv.');
end