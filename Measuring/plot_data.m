%    Copyright 2011 Nicolas Melot
%
%    Nicolas Melot (nicolas.melot@liu.se)
%
%
%    This program is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
% =============================================================================
%
%
%	Load, transform and plot the data collected through experiments, using
%	All data manipulation functions as well as custom functions. This script
%	is very dense regarding the meaning of all statements and parameters. Be
%	very carefull when modifying or writing such a file as errors may be tiny
%	but have drastic conseauences and are difficult to track.
%	

% The meaning of each columns in the matrix generated by the measurements.
% entropy nb_threads ct try thread start_time_sec start_time_nsec stop_time_sec stop_time_nsec thread_start_sec thread_start_nsec thread_stop_sec thread_stop_nsec


% First part: filtering, selection and basic transformations
collected = select(data, [1 2 3 5 6 7 8 9 10 11 12 13]); % Keep every columns except the try number.
collected = where(collected, [2], {[0 1 2 3 4 5 6 7 8]}); % Keeps only measurements involving sequential or 1, 2, 4 and 8 threads.
collected = duplicate(collected, [1 1 1 1 1 1 1 2 1 1 1 2]); % Duplicate two columns. The new columns will be used to store the computed time difference between start and stop for both global and per-thread process.
collected = apply(collected, 9, @time_difference_global); % Compute the global start-stop difference
collected = apply(collected, 14, @time_difference_thread); % Compute the start-stop time difference per threads
collected = select(collected, [1 2 3 4 9 14]); % keep every features (entropy, number of threads, number of jumps and thread number) plus the time differences calculated earlier.
collected = duplicate(collected, [1 1 1 1 2 2]); % Create 2 more columns to calculate timing standard deviations


% Second part: extraction and reshaping to produce smaller matrices
% Global timings
global_timing = groupby(collected, [1 2 3]); % Separate into groups defined by entropy, number of threads and number of loops
global_timing = reduce(global_timing, {@none, @none, @none, @none, @mean, @std, @mean, @std}); % Reduce the groups by applying mean and stadard deviation on groups' timings. Columns 1 2 and 3 are not affected as they are all identical and column four becomes meaningless, but was already because the thread number (not the number of threads) does not matter when computing the global timing.
global_timing_100 = where(global_timing, [1 3], {[0.1] [100000000]}); % Select rows denoting experiments with an entropy of 0.1 and performing 100 million jumps.
global_timing_200 = where(global_timing, [1 3], {[0.1] [200000000]}); % Select rows denoting experiments with an entropy of 0.1 and performing 200 million jumps.

% Timings per thread
thread_timing = groupby(collected, [1 2 3 4]); % Separate into groups defined by entropy, number of threads, number of loops and thread number.
thread_timing = reduce(thread_timing, {@none, @none, @none, @none, @mean, @std, @mean, @std}); % Reduce the groups by applying mean and stadard deviation on groups' timings. Columns 1 2 3 and 4 are not affected as they are all identical.
thread_timing = extend(thread_timing, [1 2 3], [4], 0); % Extend groups that do not involve the maximum amount of threads and copy the thread number to each rows of extended groups. Fills the rest with 0 (non-existent threads work for a null period of time.
thread_timing_100 = where(thread_timing, [1 3], {[0.1] [100000000]}); % Select rows denoting experiments with an entropy of 0.1 and performing 100 million jumps.
thread_timing_200 = where(thread_timing, [1 3], {[0.1] [200000000]}); % Select rows denoting experiments with an entropy of 0.1 and performing 200 million jumps.


% Third part: use plotting function to generate graphs, format and store them in graphic files.
% /!\ Matlab does not support line breaks in the middle of function calls. If you use Matlab, remove the comments and write the function call to quickplot, quickerrorbar and and quickbar in one line only.
quickplot(1,
	{global_timing_100 global_timing_200}, % Plot global timings for 100 millions and 200 millions jumps.
	2, 5, % column for x values then for y values
	{[1 0 0] [1 0 1] [0 0 1] [0 0 0] [0 0.5 0.5]}, % Colors to be applied to the curves, written in RGB vector format
	{"o" "^" "." "x" ">" "<"}, % Enough markers for 6 curves. Browse the web to find more.
	2, 15, "MgOpenModernaBold.ttf", 25, 800, 400, % Curves' thickness, markers sizes, Font name and font size, canvas' width and height
	"Number of threads", "Time in milliseconds", "Global time to perform 100 and 200 millions jumps in parallel", % Title of the graph, label of y axis and label of x axis.
	{"100m iteration, 0.1 entropy " "200m iteration, 0.01 entropy " "100m iteration, 0.00001 entropy " "300m iteration, 0.00001 entropy " }, % Labels for curves
	"northeast", "timing-error.eps", "epsc"); % Layout of the legend, file to write the plot to and format of the output file

% The two following graphs are combined together
quickerrorbar(2,
	{global_timing_100}, % Plot global timings for 100 millions jumps only.
	2, 5, 6, % column for x values then for y values and standard deviation (error bars)
	{[1 0 0] [1 0 1] [0 0 1] [0 0 0] [0 0.5 0.5]}, % Colors to be applied to the curves, written in RGB vector format
	{"o" "^" "." "x" ">" "<"}, % Enough markers for 6 curves. Browse the web to find more.
	2, 15, "MgOpenModernaBold.ttf", 25, 800, 400, % Curves' thickness, markers sizes, Font name and font size, canvas' width and height
	"Number of threads", "Time in milliseconds", "Time per thread to perform 100 millions jumps in parallel", % Title of the graph, label of y axis and label of x axis.
	{"100m iteration, 0.1 entropy "}, % Labels for curves
	"northeast", "timing-100.eps", "epsc"); % Layout of the legend, file to write the plot to and format of the output file

quickbar(2,
	thread_timing_100, % Plot global timings for 100 millions jumps only, thread by thread.
	2, 7, -1, % column for x values then for y values and base
	"grouped", 0.5, % Style of the bars ("grouped" or "stacked")
	"MgOpenModernaBold.ttf", 25, 800, 400, % Curves' thickness, markers sizes, Font name and font size, canvas' width and height
	"Number of threads", "Time in milliseconds", "Time per thread to perform 100 millions jumps in parallel", % Title of the graph, label of y axis and label of x axis.
	{"100m iteration, 0.1 entropy " "thread 1 " "thread 2 " "thread 3 " "thread 4 " "thread 5 " "thread 6 " "thread 7 " "thread 8 "}, % Labels for curves of the previous graph and bars from this graph
	"northeast", "timing-100.eps", "epsc"); % Layout of the legend, file to write the plot to and format of the output file

% The two following graphs are combined together
quickerrorbar(3,
	{global_timing_200}, % Plot global timings for 100 millions jumps only.
	2, 5, 6, % column for x values then for y values and standard deviation (error bars)
	{[1 0 0] [1 0 1] [0 0 1] [0 0 0] [0 0.5 0.5]}, % Colors to be applied to the curves, written in RGB vector format
	{"o" "^" "." "x" ">" "<"}, % Enough markers for 6 curves. Browse the web to find more.
	2, 15, "MgOpenModernaBold.ttf", 25, 800, 400, % Curves' thickness, markers sizes, Font name and font size, canvas' width and height
	"Number of threads", "Time in milliseconds", "Time per thread to perform 200 millions jumps in parallel", % Title of the graph, label of y axis and label of x axis.
	{"100m iteration, 0.1 entropy "}, % Labels for curves
	"northeast", "timing-200.eps", "epsc"); % Layout of the legend, file to write the plot to and format of the output file

quickbar(3,
	thread_timing_200, % Plot global timings for 100 millions jumps only, thread by thread.
	2, 7, -1, % column for x values then for y values and base
	"grouped", 0.5, % Style of the bars ("grouped" or "stacked")
	"MgOpenModernaBold.ttf", 25, 800, 400, % Curves' thickness, markers sizes, Font name and font size, canvas' width and height
	"Number of threads", "Time in milliseconds", "Time per thread to perform 200 millions jumps in parallel", % Title of the graph, label of y axis and label of x axis.
	{"200m iteration, 0.1 entropy " "thread 1 " "thread 2 " "thread 3 " "thread 4 " "thread 5 " "thread 6 " "thread 7 " "thread 8 "}, % Labels for curves of the previous graph and bars from this graph
	"northeast", "timing-200.eps", "epsc"); % Layout of the legend, file to write the plot to and format of the output file
