% Author: Feng Liu
% Last Modified: 2025-07-02
% Description: Get stable segments from Vicon recorded csv (with user
% overwrite function impelemented)

clear; 
close all;
filename = './filename.csv'; % Replace with your Excel file name

% Read the data from the Excel file (assuming data starts in row 1)
data = readtable(filename);

% Extract the three columns (assuming they are the first three columns)
% col1 = data{6:end, 12};  % First column
% col2 = data{6:end, 13};  % Second column
% col3 = data{6:end, 14};  % Third column

col1 = data{6:end, 6};  % First column
col2 = data{6:end, 7};  % Second column
col3 = data{6:end, 8};  % Third column

% Sum the three columns
% sum_columns = col1 + col2 + col3;
sum_columns = col1 + col2;
% sum_columns = abs(col1) + abs(col2) + abs(col3);
% sum_columns = sqrt(col1.^2 + col2.^2+ col3.^2);

% test_col = data{6:2000, 12:14};
% test_sum = sum(test_col, 2);

% Plot the summed results
figure;
plot(sum_columns, 'LineWidth', 2, 'MarkerSize', 6);
hold on;
title('Vicon Data with Highlighted Stable Segments', FontSize=18);
xlabel('Time Step (0.01 sec)', FontSize=16);
ylabel('Sum of XY Vicon Data (cm)', FontSize=16);
grid on;
ax = gca;  % Get current axes
ax.FontSize = 16;

% Identify regions where the sum does not change significantly (threshold = 1)
threshold = 0.035;
% threshold = 0.008;
stable_indices = find(abs(diff(sum_columns)) <= threshold);

% Find continuous stable segments
stable_segments = [];
if ~isempty(stable_indices)
    segment_start = stable_indices(1);
    for i = 2:length(stable_indices)
        if stable_indices(i) ~= stable_indices(i-1) + 1
            stable_segments = [stable_segments; segment_start, stable_indices(i-1)]; %#ok<AGROW>
            segment_start = stable_indices(i);
        end
    end
    stable_segments = [stable_segments; segment_start, stable_indices(end)]; % Add the last segment
end

%% check and combine segments
clean_stable_segments = [];
clean_start = 1;
seek_start_flag = true; % if true: seek what is the next start point
for i = 1 : size(stable_segments, 1)-1
    current_start = stable_segments(i, 1);
    current_end = stable_segments(i, 2);
    current_seg_length =  current_end - current_start;

    if seek_start_flag
        if current_seg_length >= 8  % avoid overly short segments as the start point
            clean_start = current_start;
            seek_start_flag = false;
        end
    end
    next_start = stable_segments(i+1, 1);

    if ~seek_start_flag && next_start - current_end >= 20  % if the gap between two segments are big, then it is the new start point
        clean_segment = [clean_start, current_end];
        clean_stable_segments = [clean_stable_segments; clean_segment];
        seek_start_flag = true;
        % clean_start = next_start;
    end

    if i==size(stable_segments, 1)-1
        clean_segment = [clean_start, stable_segments(end, 2)];
        clean_stable_segments = [clean_stable_segments; clean_segment];
    end

end

% find(abs(diff(clean_stable_segments)) <= threshold)

figure;
plot_stable_seg(sum_columns, stable_indices, clean_stable_segments);

%% stable segments amend
close all;
overwrite_segments = [1789, 1953;
                      2101, 2289;
                      12397, 12576];

if ~isempty(overwrite_segments)

for i=1:size(overwrite_segments,1)
    a1 = overwrite_segments(i, 1);
    a2 = overwrite_segments(i, 2);

    bool_overlap = (min(clean_stable_segments(:,2), a2) > max(clean_stable_segments(:,1), a1));
    if sum(bool_overlap) > 1 % overwrite input segment length too long
        disp("overwrite segments is too long, covering multiple segments!");
        break
    elseif sum(bool_overlap) == 1 % modify existed segments
        issue_segment = clean_stable_segments(bool_overlap, :);
        clean_stable_segments(bool_overlap, :) = [a1, a2];
        disp("original bad segment: ")
        disp(issue_segment)
        disp("switched to: ")

    elseif sum(bool_overlap) == 0 % add a new segments
        bool_check_tail = find(clean_stable_segments(:,2)<a1);
        row_unchange = bool_check_tail(end);
        clean_stable_segments = [clean_stable_segments(1:row_unchange, :); [a1, a2];clean_stable_segments(row_unchange+1:end, :)];
        disp("new segment added:")

    end

        disp([a1, a2])
        disp("---------------------------")

end

end

figure;
plot_stable_seg(sum_columns, stable_indices, clean_stable_segments);


function plot_stable_seg(sum_columns, stable_indices, clean_stable_segments)
    plot(sum_columns, 'LineWidth', 2, 'MarkerSize', 6);
    hold on;
    title('Vicon Data with Highlighted Stable Segments', FontSize=18);
    xlabel('Time Step (0.01 sec)', FontSize=16);
    ylabel('Sum of XY Vicon Data (cm)', FontSize=16);
    grid on;
    ax = gca;  % Get current axes
    ax.FontSize = 16;
    
    plot(stable_indices, sum_columns(stable_indices), 'bO', 'MarkerSize', 6, 'LineWidth', 2);
    hold on
    for i = 1:size(clean_stable_segments, 1)
        clean_segment_filled = clean_stable_segments(i, 1):clean_stable_segments(i, 2);
        plot(clean_segment_filled, sum_columns(clean_segment_filled), 'r*', 'MarkerSize', 6, 'LineWidth', 2);
        plot(clean_segment_filled(1), sum_columns(clean_segment_filled(1)), 'gd', 'MarkerSize', 10, 'LineWidth', 1)
    end
    % plot(clean_stable_segments, sum_columns(cleaned_stable_indices), 'r*', 'MarkerSize', 6, 'LineWidth', 2);
    hold off;
end