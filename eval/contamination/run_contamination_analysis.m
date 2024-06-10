%% EXAMPLE OF CONTAMINATION ANALYSIS

% This script analyzes the spectrogram correlations between and audio file
% and a brain recording files. Both files should be MAT-files with a
% defined structure called RecordingMatfiles (see files in 'Test data'
% folder). 'Data_preparation_example' script shows how to create these
% files.

function []=run_contamination_analysis(out_path, prepared_path, timings_path, toolbox_path)

    %% Add path to class and functions
    addpath(genpath(toolbox_path));

    %% Global parameters
    days = ["2022_09_22" "2022_09_23" "2022_09_28" "2022_09_30" "2022_10_05" "2022_10_06" "2022_10_10" "2022_10_27" "2022_11_03" "2022_11_04"];
    for i = 1:length(days)

        day = days(i);
        disp(strcat('Processing day: ', day));

        brain_matfile_path =  strcat(prepared_path, '/', day, '_KeywordReading_Overt_ecog.mat');
        audio_matfile_path =  strcat(prepared_path, '/', day, '_KeywordReading_Overt_audio.mat');
        timings_matlab_file = strcat(timings_path, '/', day, '_KeywordReading_Overt_timings.mat');
        timings = load(timings_matlab_file);
        results_path = '/tmp/Analysis results/';

        analysis_name = day;

        %% Create and store a ContaminationAnalysis object
        %
        % results_path:
        %   path to save the results
        % brain_matfile_path:
        %   brain data matfile path (should respect defined format)
        % brain_matfile_path:
        %   audio data matfile path (should respect defined format)
        % analysis_name (optional):
        %   name of files and figures related to the present analysis

        obj = ContaminationAnalysis(...
            results_path,...
            brain_matfile_path,...
            audio_matfile_path,...
            analysis_name);

        %% Select time samples that will be considered in the analysis
        %
        % select_periods:
        %   2-column array defining start and end times of the time periods to
        %   select.
        % exclude_periods:
        %   2-column array defining start and end times of the time periods to
        %   exclude.

        select_periods = []; % timings.timings;

        exclude_periods = []; % exclude the first 50 seconds

        obj = selectTime(obj,...
            select_periods,...
            exclude_periods);

        %% Detect artifacts occuring on several channels
        %
        % moving_average_span:
        %   Duration (in seconds) of the moving average window that is used to
        %   detrend the data before artifact detection.
        % artifact_threshold_factor:
        %   'artifact_threshold_factor' multiplied by the MAD of a given channel
        %   defines the artifact threshold of this channel.
        % artifact_channel_ratio:
        %   Ratio of channels crossing their threshold for a sample to be
        %   considered as an artifact
        % artifact_safety_period:
        %   Period of time (in seconds) before and after artifact in which samples
        %   are also considered as artifacts

        moving_average_span = 0.5;
        artifact_threshold_factor = 5;
        artifact_channel_ratio = 1/10;
        artifact_safety_period = 0.5;

        obj = detectArtifacts(obj,...
            moving_average_span,...
            artifact_threshold_factor,...
            artifact_channel_ratio,...
            artifact_safety_period);

        %% Display the results of the artifact detection and save the figure
        %
        % display_channel_nb:
        %   Number of channels to show. The first half of the displayed channels
        %   are the channels with the highest numbers of artifact samples and the
        %   second half are the ones with the lowest numbers.
        %
        % Can return figure handle.

        display_channel_nb = 6;

        % displayArtifacts(obj, display_channel_nb)

        %% Compute the spectrograms of the audio and brain recordings
        %
        % window_duration:
        %   Duration of the spectrogram window (in seconds).
        % spg_fs:
        %   Desired sampling frequency of the spectrogram.
        % spg_freq_bounds:
        %   2-element vector containing the lowest and the highest frequencies
        %   considered in the spectrogram (if empty, all frequency bins are kept).

        window_duration = 200e-3;
        spg_fs = 50;
        spg_freq_bounds = [70 170];

        obj = computeSpectrograms(obj,...
            window_duration, spg_fs,spg_freq_bounds);

        %% Compute spectrogram correlations between the audio and the brain data

        obj = computeSpectrogramCorrelations(obj);

        %% Display the spectrogram correlations and save the figures
        %
        % disp_freqs_bounds:
        %   2-element vector containing the lowest and the highest frequencies
        %   displayed in the spectrogram (if empty, all frequency bins are kept).
        % display_channels:
        %   'index' or 'id' of the channels to be displayed.
        % colormap_limits:
        %   2-element vector containing the lowest and the limits of the colormap
        %   displaying the z-scored spectrograls.
        %
        % Can return figure handles.

        display_channels = [];
        disp_freqs_bounds = [];
        colormap_limits = [0 5];

        % displayCorrelations(obj, disp_freqs_bounds, display_channels, colormap_limits);

        %% Compute spectrogram cross-correlations between the audio and the brain data
        %
        % max_time_lag:
        %   Maximum absolute time lag in seconds considered when applying positive
        %   and negative delays to the audio spectrogram.

        max_time_lag = 0.5;

        obj = computeSpectrogramCrossCorrelations(obj, max_time_lag);


        %% Display cross-correlations
        %
        % crosscorr_min_max_freqs:
        %   2-element vector containing the lowest and the highest
        %   frequencies to be considered (if empty, all frequency bins are kept).
        % top_corr_ratio:
        %   Ratio of the highest cross-correlograms to display. 0.01 means that the
        %   1% of cross-correlograms reaching the highest values will be displayed.

        crosscorr_min_max_freqs = [70 170]; % frequency range considered
        top_corr_ratio = 0.01; % ratio of the highest correlations to display

        % displayCrossCorrelations(obj, crosscorr_min_max_freqs, top_corr_ratio)


        %% Compute statistical criterion P
        %
        % criterion_min_max_freqs:
        %   2-element vector containing the lowest and the highest
        %   frequencies to be considered (if empty, all frequency bins are kept).

        criterion_min_max_freqs = [70 170];

        obj = computeStatisticalCriterion(obj, criterion_min_max_freqs);

        %% Display statistical criterion P

        % displayStatisticalCriterion(obj);

        out.surrogate_measures = obj.surrogate_measures;
        out.dataset_measure = obj.dataset_measure;
        out.criterion_value = obj.criterion_value;
        save(strcat(out_path, '/', day, '_contamination_result.mat'),'out');
        clear out;
    end
end
