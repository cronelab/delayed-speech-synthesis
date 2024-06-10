function []=data_preparation(out_path, content_path, toolbox_path)

    %% Add path to class and functions
    folder_content = dir(content_path);
    addpath(genpath(toolbox_path));

    nb_files = length(folder_content);
    for i = 1:nb_files
        file = folder_content(i);
        if file.isdir == 1
            continue;
        end

        [filepath, name, ext] = fileparts(file.name);

        %% Output file path
        output_path_audio = strcat(out_path, '/', name, '_audio.mat');
        output_path_ecog = strcat(out_path, '/', name, '_ecog.mat');

        %% Load a sample keyword recording
        f = load(strcat(file.folder, '/', file.name));

        %% Prepare audio data

        % Export to required file format
        createRecordingMatfile(output_path_audio, f.audio', double(f.fs));


        %% Prepare ecog data

        % Export to required file format
        createRecordingMatfile(output_path_ecog, f.ecog, double(f.fs));
    end
end