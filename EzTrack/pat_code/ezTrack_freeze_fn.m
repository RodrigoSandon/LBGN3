function [data] = ezTrack_freeze_fn(data, ezTrack_filename_strings, ezTrack_data, CS_frames, frames)

uv.MotionCutoff = ezTrack_data.MotionCutoff(1);
uv.FreezeThresh = ezTrack_data.FreezeThresh(1);
uv.MinFreezeDuration = ezTrack_data.MinFreezeDuration(1);

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
                    filtered_freezing{good_index} =ezTrack_data.Freezing(frames(onset+2:offset+2)); %SLEAP_data.vel_cm_s(SLEAP_data.idx_frame(onset:offset));
                    filtered_motion{good_index} =ezTrack_data.Motion(frames(onset+2:offset+2)); %SLEAP_data.vel_cm_s(SLEAP_data.idx_frame(onset:offset));
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