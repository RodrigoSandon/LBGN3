%% Requirements: 
% ABET2TableFn_Chamber_A_v6: reorganizing ABET data as a table
% TrialFilter:  filter ABET table to just trials that you want to look at,
%               based on event type
% dbmFormatInputs: from Zhang et al., 2021
% dbmTrainModel: from Zhang et al., 2021
% dbmGetModelOutputs: from Zhang et al., 2021
% dbmExtractMicrostates: from Zhang et al., 2021

%% Load formatted data for DBM
SLEAP_data = readtable('67_RDT_D1_body_sleap_data.csv');
SLEAP_x_data = readtable('67_RDT_D1_x_motion.csv');
SLEAP_y_data =  readtable('67_RDT_D1_y_motion.csv');


%% Edit these uservariables with what you want to look at
uv.evtWin = [-10 30]; %what time do you want to look at around each event [-10 10] [-5 35]
uv.BLper = [-10 -5]; %what baseline period do you want for z-score [-10 -5] [-5 0]
uv.dt = 0.3; %what is your frame rate 0.2
uv.behav = 'choiceTime'; %which behavior/timestamp to look at choiceTime

%% Load behavioral data and adjust the timestamps
[BehavData,ABETfile,Descriptives, block_end]=ABET2TableFn_Chamber_A_v6('67 10232019.csv',[]);

ABET_removeheader = ABETfile(2:end,:);

tbl_ABET = cell2table(ABET_removeheader);
tbl_ABET.Properties.VariableNames = ABETfile(1,:);

%% Adjust timestamps (necessary for photometry - camera starts with Synapse recording, then behavior starts some time after. Can detect this from the video, or from the recording epoch.)
timeStart = 6.170; %ABET start time (from video or from the TDT epoch)

%align timestamps of behavioral data to timestamps of photometry by adding
%the time elapsed from when Synapse recording began to when the ABET
%program was started back to each relevant time column (choiceTime,
%collectionTime, stTime)
timeShift=timeStart*ones(numel(BehavData.choiceTime(:)),1);
% timeShift_SLEAP = timeStart*ones(size(SLEAP_data));
% SLEAP_data.idx_time = SLEAP_data.idx_time+timeStart;
BehavData.choiceTime(:)=BehavData.choiceTime(:)+timeShift;
BehavData.collectionTime(:)=BehavData.collectionTime(:)+timeShift;
BehavData.stTime(:)=BehavData.stTime(:)+timeShift;
block_end = block_end +timeShift;
% shk_times(:)=shk_times(:)+stTime(1);

v = VideoReader('67_RDT_D1.2019-10-23T13_48_06.avi');

%% Inscopix-specific details
gpio_tbl = readtable('BLA-Insc-1_2021-01-20_RDT_D1_GPIO.csv');

shk_times = tbl_ABET.Evnt_Time(strcmp(tbl_ABET.Item_Name, 'shock_on_off') & tbl_ABET.Arg1_Value == 1);

stTime = gpio_tbl.Time_s_(strcmp(gpio_tbl.ChannelName, 'GPIO-2') & gpio_tbl.Time_s_ > 0);

frames = gpio_tbl.Time_s_(strcmp(gpio_tbl.ChannelName,'BNC Sync Output') & gpio_tbl.Value == 1);

%check GPIO file to extract each TTL, since the TTL is 1000ms and is
%sampled repeatedly. This will only extract events that are separated by >
%8sec, so be sure to change this if the TTL or task structure changes
%dramatically! 
pp = 2;
ttl_filtered = stTime(1);
for kk = 1:size(stTime,1)-1
    if abs(stTime(kk)-stTime(kk+1)) > 8
        ttl_filtered(pp) = stTime(kk+1);
        pp=pp+1;
    end
end
ttl_filtered = ttl_filtered';      

%Add TTL times received by Inscopix to data table, skipping omitted trials
%which do not have a corresponding TTL due to a quirk in the behavioral
%program
BehavData.Insc_TTL = zeros(length(BehavData.TrialPossible),1);
dd = 2;
for cc = 1:size(BehavData, 1)
    if BehavData.TrialPossible(cc) > stTime(1)
        BehavData.Insc_TTL(cc) = ttl_filtered(dd);
        dd = dd+1;
    elseif BehavData.TrialPossible(cc) <= stTime(1)
        BehavData.Insc_TTL(cc) = 0;
    end
end

BehavData.choTime2 = BehavData.choiceTime-BehavData.TrialPossible;
BehavData.choTime3 = BehavData.Insc_TTL+BehavData.choTime2;

%%
%filter based on TrialFilter inputs (see TrialFilter.m for full list of
%possibilities)
BehavData=TrialFilter(BehavData,'OMITALL',0);


% get times to different events - be sure to change this if aligning to
% START vs CHOICE etc. 
time2Collect = BehavData.collectionTime(:)-BehavData.stTime(:);
time2Choice = BehavData.choiceTime(:)-BehavData.stTime(:);

large_choice = BehavData.bigSmall == 1.2;
small_choice = BehavData.bigSmall == 0.3;

[numTrials,~]=size(BehavData.collectionTime(:));
Tris=[1:numTrials]';

%%

% get time windows around choice
% time_ranges(1,:) = BehavData.choiceTime + uv.evtWin(1);
% time_ranges(2,:) = BehavData.choiceTime + uv.evtWin(2);

% get time windows around start (this seems safer and more in line with the
% DBM paper from Da-Ting's lab (Zhang & Denman et al. 2022)
time_ranges(1,:) = BehavData.stTime + uv.evtWin(1);
time_ranges(2,:) = BehavData.stTime + uv.evtWin(2);


fs_cam = 30;

% create time series for the size of the window, based on sampling rate of
% camera
ts1 = uv.evtWin(1):1/(fs_cam):uv.evtWin(2);

if ~isempty(SLEAP_x_data)
    n = 2;
    for i = 1:length(n)
        fs_cam = 30; %set sampling rate according to camera, this is hard coded for now
        filtered = [];
        max_ind = max(size(SLEAP_x_data));
        good_index = 1;
        for j = 1:size(time_ranges,2)
            onset = round(time_ranges(1,j)*fs_cam)+1;
            offset = round(time_ranges(2,j)*fs_cam)+1;
            % throw it away if onset or offset extends beyond recording window
            if isinf(offset)
                if onset <= max_ind && onset > 0
                    filtered{good_index} = SLEAP_data(onset:end);
                    break %return
                end
            else
                if offset <= max_ind && offset > 0 && onset <= max_ind && onset > 0


                    body_parts_X_test{j, 1}(1:6,:) = table2array(SLEAP_x_data(onset:offset,2:7))';
                    body_parts_Y_test{j, 1}(1:6,:) = table2array(SLEAP_y_data(onset:offset,2:7))';
% 

                    %create "blank" categorical cell array to fill in with pseudolabels
                    behavior_pseudolabels_test{j, 1} = categorical(ones(1,size(ts1,2)));
                    good_index = good_index + 1;
                end
            end
        end
%         if KEEPDATA
%             data.streams.Motion.filtered = filtered;
%         else
%             data.streams.Motion.data = filtered;
%             data.streams.Motion.filtered = [];
%         end
    end
end



%%

arena_zones = struct;

possible_labels = categorical({'LOCOMOTION', 'LEFT_SCREEN', 'RIGHT_SCREEN', 'FOOD_CUP', 'OTHER'});
pseudolabel_cats = categories(possible_labels);

frame = read(v,1000);
imtool(frame) 
% h = images.roi.Freehand(gca,'Position',[100 150;200 250;300 350;150 450]);


% would be really nice to build a UI based approach to labeling, but having
% trouble communicating b/w the UI and the Workspace via Callback w/
% "selection" function.
% imshow(frame);
% g = uicontrol('Style','popupmenu','String', possible_labels,'Position',[20 20 100 20]);
% l = uicontrol('Style','pushbutton','String','Done','Position',[200 20 100 20]);
% g.Callback = @selection;
% % l.Callback = @myCloseReq;


imshow(frame);
disp('Draw border around left-lever zone')
left_zone = drawpolygon('StripeColor','r');
left_max = max(left_zone.Position);
left_min = min(left_zone.Position);
input('Confirm when left-lever zone is drawn (Y): \n','s');

disp('Draw border around right-lever zone')
right_zone = drawpolygon('StripeColor','y');
right_max = max(right_zone.Position);
right_min = min(right_zone.Position);
input('Confirm when right-lever zone is drawn (Y): \n','s');

disp('Draw border around foodcup zone')
foodcup_zone = drawpolygon('StripeColor','b');
foodcup_max = max(foodcup_zone.Position);
foodcup_min = min(foodcup_zone.Position);
input('Confirm when foodcup zone is drawn (Y): \n','s');

%%
for kk = 1:size(body_parts_X_test,1)

    left_screen_array(kk,:) = ((body_parts_X_test{kk, 1}(1,:)) < left_max(1, 1) & (body_parts_X_test{kk, 1}(1,:)) > left_min(1, 1)) & ((body_parts_Y_test{kk, 1}(1,:)) < left_max(1, 2)) & (body_parts_Y_test{kk, 1}(1,:)) > left_min(1, 2);
    right_screen_array(kk,:) = ((body_parts_X_test{kk, 1}(1,:)) < right_max(1, 1) & (body_parts_X_test{kk, 1}(1,:)) > right_min(1, 1)) & ((body_parts_Y_test{kk, 1}(1,:)) < right_max(1, 2)) & (body_parts_Y_test{kk, 1}(1,:)) > right_min(1, 2);
    food_cup_array(kk,:) = ((body_parts_X_test{kk, 1}(1,:)) < foodcup_max(1, 1) & (body_parts_X_test{kk, 1}(1,:)) > foodcup_min(1, 1)) & ((body_parts_Y_test{kk, 1}(1,:)) < foodcup_max(1, 2)) & (body_parts_Y_test{kk, 1}(1,:)) > foodcup_min(1, 2);
    others = left_screen_array(kk,:) + right_screen_array(kk,:) + food_cup_array(kk,:);
    other_array(kk,:) = others == 0; 


    behavior_pseudolabels_test_2{kk}(1, left_screen_array(kk,:)) = possible_labels(2);
    behavior_pseudolabels_test_2{kk}(1, right_screen_array(kk,:)) = possible_labels(3);
    behavior_pseudolabels_test_2{kk}(1, food_cup_array(kk,:)) = possible_labels(4);
    behavior_pseudolabels_test_2{kk}(1, other_array(kk,:)) = possible_labels(5);

end

behavior_pseudolabels_test_2 = behavior_pseudolabels_test_2';

%%
pose_sequences = dbmFormatInputs(body_parts_X_test, body_parts_Y_test);

%% Train DBM model

N = numel(pose_sequences);

use_gpu = false;
val_fraction = 0.1; % Proportion of input sequences to be held back as validation data

val_ind = false(N,1);
val_ind(randperm(N, round(N*val_fraction))) = true;


[dbm_mdl, dbm_train_report] = dbmTrainModel(pose_sequences(~val_ind), behavior_pseudolabels_test_2(~val_ind), ... % training data
    pose_sequences(val_ind), behavior_pseudolabels_test_2(val_ind), use_gpu); % validation data

% dbm_mdl = the trained DBM network, which can be used to predict
%   behavioral pseudolabels and/or map pose sequences to the latent space

% dbm_train_report = summary data about how the model performed
%   during the training process.

%%

[pseudolabel_predictions, latent_trajectories] = dbmGetModelOutputs( dbm_mdl, pose_sequences, use_gpu );

% pseudolabel_predictions = model's prediction of original training targets
% latent trajectories = set of N (default = 10) variables mapping subject's
%   behavior to a point in the "behavior space" learned by the model

%% Extract microstates using k-means algorithm

num_microstates = 50;
max_iter = 100;
replicates = 1;
use_parallel = false;

% Extract microstates
[microstate_labels, microstate_centroids] = dbmExtractMicrostates(latent_trajectories, num_microstates, max_iter, replicates);

% Re-numebr microstates according to which pseudolabel category they
% overlap with most

microstates = [1:num_microstates];
label_overlap = (cat(2,behavior_pseudolabels_test_2{:})==pseudolabel_cats) * (cat(2,microstate_labels{:}).'==microstates);
[~,max_overlap] = max(label_overlap,[],1);
[~,sort_ind] = sortrows([max_overlap; sum(label_overlap,1)].',[1 2]);
microstates = microstates(sort_ind);
label_overlap = label_overlap(:,sort_ind);
[~,microstate_labels] = cellfun(@(X) ismember(X,microstates), microstate_labels, 'UniformOutput', false);

% Normalize overlap matrix column-wise
label_overlap = round(1000.*(label_overlap./sum(label_overlap,1)))./10;


%% Visualization
figure(1)
clf

h = heatmap(label_overlap);
h.XDisplayLabels = string(microstates);
h.YDisplayLabels = pseudolabel_cats;
title({'% pseudolabel category by microstate'; '(columns sum to ~100%)'})
xlabel('Microstate #')
ylabel('Behavior pseudolabel category')


% figure(2)
% clf
% 
% imagesc(cat(1,microstate_labels{:}));
% colormap(jet(num_microstates))
% cb = colorbar;
% ticks = linspace(cb.Limits(1), cb.Limits(2), num_microstates+1);
% ticks = ticks(1:end-1) + mean(diff(ticks))/2;
% cb.Ticks = ticks;
% cb.TickLabels = 1:num_microstates;
% ylabel('Trials')
% xlabel('Video frames')
% title('Microstate occurrence')

figure(2)
clf

imagesc(ts1, 1, (cat(1,microstate_labels{:}))); hold on;
colormap(jet(num_microstates))
cb = colorbar;
scatter(time2Collect,Tris,'Marker','o','MarkerFaceColor','g','SizeData',30,'MarkerEdgeColor','black')
scatter(time2Choice(large_choice),Tris(large_choice),'Marker','s','MarkerFaceColor','w','SizeData',30,'MarkerEdgeColor','black')
scatter(time2Choice(small_choice),Tris(small_choice),'Marker','s','MarkerFaceColor',[96 96 96]/255,'SizeData',30,'MarkerEdgeColor','black')
plot(zeros(numTrials,1),Tris,'Color','white', 'LineStyle', ':', 'LineWidth', 2)
ticks = linspace(cb.Limits(1), cb.Limits(2), num_microstates+1);
ticks = ticks(1:end-1) + mean(diff(ticks))/2;
cb.Ticks = ticks;
cb.TickLabels = 1:num_microstates;
ylabel('Trials')
xlabel('Time from trial start (s)')
title('Microstate occurrence')
hold off;

figure(3)
clf
imagesc(ts1, 1,double(cat(1,behavior_pseudolabels_test_2{:}))); hold on;
colormap(parula(numel(pseudolabel_cats)))
cb = colorbar;
scatter(time2Collect,Tris,'Marker','o','MarkerFaceColor','g','SizeData',30,'MarkerEdgeColor','black')
scatter(time2Choice(large_choice),Tris(large_choice),'Marker','s','MarkerFaceColor','w','SizeData',30,'MarkerEdgeColor','black')
scatter(time2Choice(small_choice),Tris(small_choice),'Marker','s','MarkerFaceColor',[96 96 96]/255,'SizeData',30,'MarkerEdgeColor','black')
plot(zeros(numTrials,1),Tris,'Color','white', 'LineStyle', ':', 'LineWidth', 2)

ticks = linspace(cb.Limits(1), cb.Limits(2), numel(pseudolabel_cats)+1);
ticks = ticks(1:end-1) + mean(diff(ticks))/2;
cb.Ticks = ticks;
cb.TickLabels = pseudolabel_cats;
title('Pseudolabel occurrence')
ylabel('Trials')
xlabel('Time from trial start (s)')

%% try to plot trajectories based on the 5s before choice all the way to collection
%still in progress
for ii = 1:size(Tris,1)
    trial_length_ind(ii,:) =  ts1 > (time2Choice(ii) - 5) & ts1 < time2Collect(ii)
end

% test = latent_trajectories{1, 1}(trial_length_ind(1,:))