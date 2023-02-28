

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
disp(CS_on)

CS_frames = [CS_on*freezing_samples_per_sec CS_off*freezing_samples_per_sec];

ezTrack_file = '/media/rory/Padlock_DT/Fear_Conditioning_Control/NewVideos/Conditioning/RRD287_Extinction_FreezingOutput.csv';
ezTrack_filename_strings = split(ezTrack_file, '_');

ezTrack_data = readtable(ezTrack_file);

frame_filter = 'Frame';
frames = ezTrack_data.(frame_filter);

data = struct;
uv = struct;
uv.MotionCutoff = ezTrack_data.MotionCutoff(1);
uv.FreezeThresh = ezTrack_data.FreezeThresh(1);
uv.MinFreezeDuration = ezTrack_data.MinFreezeDuration(1);
%%
KEEPDATA  = 1;
if ~isempty(ezTrack_data)
%     n = fieldnames(data.streams);
%     for i = 1:length(n)
%         fs_cam = 120; %set sampling rate according to camera, this is hard coded for now
        filtered = [];
        max_ind = max(frames);
        good_index = 1;
        for j = 1:size(CS_frames,1)
            onset = CS_frames(j,1);
            offset = CS_frames(j,2);
            % throw it away if onset or offset extends beyond recording window
            if isinf(offset)
                if onset <= max_ind && onset > 0
                    filtered{good_index} = SLEAP_data(onset:end);
                    break %return
                end
            else
                if offset <= max_ind && offset > 0 && onset <= max_ind && onset > 0
                    filtered_freezing{good_index} =ezTrack_data.Freezing(frames(onset+2:offset+2)); %onset and offset are adjusted because the ezTrack_data Table has a header and starts at frame 0
                    filtered_motion{good_index} =ezTrack_data.Motion(frames(onset+2:offset+2)); %onset and offset are adjusted because the ezTrack_data Table has a header and starts at frame 0
                    good_index = good_index + 1;
                end
            end
        end
        if KEEPDATA
            data.streams.(ezTrack_filename_strings{1}).(ezTrack_filename_strings{2}).Freezing = filtered_freezing;
            data.streams.(ezTrack_filename_strings{1}).(ezTrack_filename_strings{2}).Motion = filtered_motion;
            data.streams.(ezTrack_filename_strings{1}).(ezTrack_filename_strings{2}).uv = uv;
%             data.streams.Motion.filtered = filtered;
        else
%             data.streams.Motion.data = filtered;
%             data.streams.Motion.filtered = [];
%         end
    end
end

%%
allSignals_freezing = cell2mat(data.streams.(ezTrack_filename_strings{1}).(ezTrack_filename_strings{2}).Freezing)';
allSignals_motion = cell2mat(data.streams.(ezTrack_filename_strings{1}).(ezTrack_filename_strings{2}).Motion)';


ts1 = 0:1/freezing_samples_per_sec:floor(size(allSignals_freezing, 2)/30);

first_5_CS_freezing = mean(allSignals_freezing(1:5, :));
first_5_CS_motion = mean(allSignals_motion(1:5, :));

figure; plot(ts1, mean(allSignals_freezing(1:5, :), 1));
figure; plot(ts1, mean(allSignals_motion(1:5, :), 1));
figure; plot(1:CS_num, mean(allSignals_freezing, 2))
