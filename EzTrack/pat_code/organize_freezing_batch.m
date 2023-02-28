CS_num = 50; %number of CS's
freezing_samples_per_sec = 30;

% code created by ChatGPT with the following prompt: Create a MATLAB code that creates variables CS_on and CS_off. There are 50 CS's each lasting 30 seconds. The first CS occurs at 180 seconds. Each CS is separated by 5 seconds. Store the times (in seconds) that the CS comes on in CS_on and that the CS turns off in CS_off. 
CS_on = zeros(CS_num,1); % create an array to store CS on times
CS_off = zeros(CS_num,1); % create an array to store CS off times

start_time = 180; % start time for first CS
interval = 5; % interval between each CS
duration = 30; % duration of each CS

for i = 1:50
CS_on(i) = start_time + (i-1)*(interval + duration);
CS_off(i) = CS_on(i) + duration;
end

CS_num_conditioning = 3;
CS_on_conditioning = zeros(CS_num_conditioning, 1);
CS_off_conditioning = zeros(CS_num_conditioning, 1);
interval_conditioning = [0, 60, 75]; %interval between each CS during conditioning; because of the math below, have to use 75 for the last interval

% q = 1;
for i = 1:3
    CS_on_conditioning(i) = start_time + (i-1)*(interval_conditioning(i)+duration);
    CS_off_conditioning(i) = CS_on_conditioning(i) + duration;
%     q = q+1;
end

CS_frames_conditioning = [CS_on_conditioning*freezing_samples_per_sec CS_off_conditioning*freezing_samples_per_sec];
CS_frames_extinction = [CS_on*freezing_samples_per_sec CS_off*freezing_samples_per_sec];
CS_frames_retrieval = [CS_on(1:5)*freezing_samples_per_sec CS_off(1:5)*freezing_samples_per_sec];

data = struct;
uv = struct;


%%
%Put all ezTrack files in to one directory, copy path to directory below
myfiles = dir('H:\Risk\Data\Fear Conditioning Control\NewVideos\FC Control All ezTrack Files');

%Get the filenames and folders of all files and folders inside the folder
%of your choice.
filenames={myfiles(:).name}';
filefolders={myfiles(:).folder}';
%Get only those files that have a csv extension and their corresponding
%folders.
csvfiles=filenames(endsWith(filenames,'.csv'));
csvfolders=filefolders(endsWith(filenames,'.csv'));

%Make a cell array of strings containing the full file locations of the
%files.
files=fullfile(csvfolders,csvfiles);

%%
for c = 1:size(files, 1)
    ezTrack_file = csvfiles{c};
    ezTrack_filename_strings = split(ezTrack_file, '_');
    ezTrack_data = readtable(ezTrack_file);
    frame_filter = 'Frame';
    frames = ezTrack_data.(frame_filter);

    if contains(ezTrack_file, 'Conditioning')
        
        CS_frames = CS_frames_conditioning;
        [data] = ezTrack_freeze_fn(data, ezTrack_filename_strings, ezTrack_data, CS_frames, frames);
    elseif contains(ezTrack_file, 'Extinction')
        ezTrack_data = readtable(ezTrack_file);
        CS_frames = CS_frames_extinction;
        [data] = ezTrack_freeze_fn(data, ezTrack_filename_strings, ezTrack_data, CS_frames, frames);
    elseif contains(ezTrack_file, 'Retrieval')
        ezTrack_data = readtable(ezTrack_file);
        CS_frames = CS_frames_retrieval;
        [data] = ezTrack_freeze_fn(data, ezTrack_filename_strings, ezTrack_data, CS_frames, frames);
    end

end


