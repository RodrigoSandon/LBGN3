function [data, ABETdata, Descriptives, block_end] = ABET2TableFn_Chamber_A_v6(filename, dummy)


%ABET2Table creates a table with columns (outlined under "column headers"
%comment) from ABET behavioral data "filename"

%ABET file should be either Raw data, or reduced data so long as all
%relevant events are included
%input variable should be the file name, written in single quotation marks.
% 3/29/2020: Updated the omission filtering such that it can now also be
% filtered by Block or by Type (Force/Free)

% Added "TrialPossible" which looks for TTL#1 (or whatever it is named,
% redo this if there are multiple TTL#s), which indicates the increment of
% every new trial (and when ABET sends a TTL to the Inscopix system)

%initiate table

%column headers
% (1)Trial: trial number
% (2)Block: 1,2,3,4, or 5
% (3)Force/Free: 0 if free, 1 if force
% (4)big/small:
% (5)stTime: trial start time
% (6)choicetime: timestamp of choice 
% (7)collection time: timestamp of reward collection
% (8)shock: 0 if no shock, 1 if shock
% (9)omission: 0 if false, 1 if free trial omission, 2 if forced trial
% (10) omissionALL: 1 for omissions, can now also filter by Block and
% Force/Free
% (11)WL: 1 if win; 3 if loss
% (12)WSLScode: 2 if win+1; 4 if loss+1;O
% (13)win_stay: 1 for large/risky choice trials following a large/risky
% choice with no punishment ("win)
% (14)lose_shift: 1 for safe choice trials following a large/risky choice
% with punishment ("loss")
% (15)lose_omit: 1 for omissions following a large/risky choice with
% punishment ("loss")
% (16)smallRew: total FREE CHOICE smallRew trials
% (17)bigRew: total FREE CHOICE bigRew trials

[ABETdata]=readcell(filename);

Headers={'Trial','Block','ForceFree','bigSmall','RewSelection','TrialPossible','stTime','choiceTime'...
    'collectionTime','shock','omission','omissionALL','WL','WSLScode','win_stay','lose_shift','lose_omit','smallRew','bigRew'};
data=table(zeros(80,1),zeros(80,1),zeros(80,1),zeros(80,1),zeros(80,1),zeros(80,1),zeros(80,1),zeros(80,1),zeros(80,1),zeros(80,1),zeros(80,1),zeros(80,1),zeros(80,1),zeros(80,1),zeros(80,1),zeros(80,1),zeros(80,1),zeros(80,1),zeros(80,1));
data.Properties.VariableNames([1:19])=Headers;

%%
%add ABET data to table

%loop through all rows
[rows,~]=size(ABETdata);
trial=1;
blocks = 2;
block_end = [];
tbl_size = [];

%find the first nonzero timestamp (all timestamps at 0 are program checks,
%and we don't care about these when we're searching for behavioral events
stop=999; rr=2;
while stop>0
    startRow=rr;
    if ABETdata{rr,1}>0
        stop=-999;
    end
    rr=rr+1;
    
    
end

currBlock=1;

for ii=startRow:rows
    
    
    
    data.Trial(trial)=trial;
    
    %keep track of block
    if regexp(ABETdata{ii,4},'Session*')
        if ABETdata{ii,9}==1
            whichBlock=regexp(ABETdata{ii,4},'n','split'); %pull block from "SX-Free...=big" or similar name
            currBlock=str2num(whichBlock{2});

        end
        
    end
    
    data.Block(trial)=currBlock;
    
    %this was added to add Blocks to BORIS data. Plan to add a check for
    %the BORIS timestamp for an Approach/Abort, to see if the value is <
    %block_end(1,1), which corresponds to Block 1, > block_end(1,1) and <
    %block_end(1,2), which corredsponds to Block 2, and > block_end(1,2),
    %which corresponds to Block 3
    if blocks == 2 &&  data.Block(trial) == 2
        block_end(1,1) = ABETdata{ii,1};
        blocks = blocks + 1;
    elseif blocks == 3 &&  data.Block(trial) == 3
        block_end(1,2) = ABETdata{ii,1};
        blocks = blocks + 1;
    end
    
    %COLLECTION TIME
    %because I increment the trial based on feeder, add each
    %reward retrieved to the previous trial
    if regexp(ABETdata{ii,4},'Reward Retrieved*')
        data.collectionTime(trial-1)=ABETdata{ii,1};
    end
    
    %TrialStart
    if regexp(ABETdata{ii,4},'\w*Trials Begin','ignorecase')
        data.stTime(trial)=ABETdata{ii,1};
    end
    
    if regexp(ABETdata{ii,4},'TTL\w*')
        data.TrialPossible(trial)=ABETdata{ii,1};
    end
                
    %CHOICE TIME
    %if it's a choice
    if ABETdata{ii,2}==1 && ~ischar(ABETdata{ii,6})
        if ((ABETdata{ii,6}==9) || (ABETdata{ii,6}==12))
            if regexp(ABETdata{ii,4},'S*')
                touch=regexp(ABETdata{ii,4},'-','split');
              
                
                
                %force or free?
                if strcmp(touch{2},'Free')
                    data.ForceFree(trial)=0;
                elseif strcmp(touch{2},'Forced')
                    data.ForceFree(trial)=1;
                end
                
                %choice time
                data.choiceTime(trial)=ABETdata{ii,1};
                

            end
            
        end
        
         
    end
    
    %REWARD DELIVERY
    %if it's feeder, store delivery duration in bigSmall column, then
    %increment trial by 1
    %5/5/2021
    %added some details because for some mice the SMALL REW was 0.3 or
    %0.5s, depending on how fast the particular feeder spun. To make it
    %easier to work with, I'm converting all 0.5s to 0.3 (so that
    %TrialFilter etc. continues to work without being re-written)
    
    if strcmp(ABETdata{ii,4},'Feeder #2')
        data.bigSmall(trial)=ABETdata{ii,9};
        if data.bigSmall(trial) == 0.5
            data.bigSmall(trial) = 0.3;
%         elseif data.bigSmall(trial) == 0.3 || 0.5
%             data.RewSelection(trial) = 2;
        end
    end
    
    
    %if there's a shock
    %if there's a shock, the shock is recorded as happening after the most
    %recent reward delivery.  So,the trial counter will have already
    %incremeneted by 1.  In order to properly record the shock trial, set
    %shock to 1 on the nth trial, where n=trial-1.
    if strcmp(ABETdata{ii,4},'Shocker #1')
        data.shock(trial-1)=1;
        
    end
    
    
    
    if data.bigSmall(trial)~=0
        trial=trial+1;
        data.Trial(trial)=trial;
    end
    
    
    
    %if it's an omission --right now, omission trials aren't labeled within
    %their blocks.
%     if strcmp(ABETdata{ii,4},'freetrial_omission')
%         data.omission(trial)=1;
%         data.choiceTime(trial)=ABETdata{ii,1};
%         trial=trial+1;
%     elseif strcmp(ABETdata{ii,4},'forcedtrial_omission')
%         data.omission(trial)=2;
%         data.choiceTime(trial)=ABETdata{ii,1};
%         trial=trial+1;
%     end
    
    %if its an omission regardless of forced or free (has to be lowercase
    %because other indicators of omission are in uppercase)
%      if regexp(ABETdata{ii,4},'\w*omission')
%         data.omissionALL(trial)=1;
%         data.choiceTime(trial)=ABETdata{ii,1};
%         omits=regexp(ABETdata{7502,4},'_','split');
%         trial=trial+1;
%      end
     
    if regexp(ABETdata{ii,4},'Omission of a*')
        data.omissionALL(trial)=1;
        data.bigSmall(trial)=999;
        data.choiceTime(trial)=ABETdata{ii,1};
        omit_str=regexp(ABETdata{ii,4},' ','split');
        
         %force or free?
                if strcmp(omit_str{4},'Free')
                    data.ForceFree(trial)=0;
                elseif strcmp(omit_str{4},'Forced')
                    data.ForceFree(trial)=1;
                end
%         if blocks == 2 && currBlock == 2
%             block_end(1,1) = data.choiceTime(trial);
%             blocks = blocks + 1;
%         elseif blocks == 3 && currBlock == 3
%             block_end(1,2) = data.choiceTime(trial);
%         end
   
            
        trial=trial+1;
    end
    %if its an ITI state
    %if strcmp(ABETdata{ii,4},'ITI_timer'
end

%Last "Trial" in data table is not a complete trial - delete this trial
%from table
data(max(data.Trial),:) = [];

%add win stay/lose shift info.  To do this, add a column for
%win-stay/lose-shift code, called WSLS code.  For this code,
% if trial is a win, code=1;
% if trial is the trial after a win, code=2;
% if trial is a loss, code=3;
% if trial is a trial after a loss, code=4;




for jj=1: numel(data.Trial)

    if data.omission(jj)==0
        if data.bigSmall(jj)== 1.2 && data.ForceFree(jj)==0 %if data.bigSmall(jj)==1.2 && data.ForceFree(jj)==0
            data.bigRew(jj)=1;
            if data.shock(jj)==0
                data.WL(jj)=1; %win
            elseif data.shock(jj)==1
                data.WL(jj)=3; %loss
            end
        end
        if data.bigSmall(jj)== 0.3 && data.ForceFree(jj)==0 %if data.bigSmall(jj)==0.3 && data.ForceFree(jj)==0 % || data.bigSmall(jj)==0.5
                   data.smallRew(jj)=1;       
        end
        if jj>1
            if data.WL(jj-1)==1
                data.WSLScode(jj)=2; %win+1 trial
                if data.bigSmall(jj)== 1.2 && data.ForceFree(jj)==0 %if data.bigSmall(jj)==1.2 && data.ForceFree(jj)==0
                    data.win_stay(jj)=1; %win_stay is 1 if chose big after a win
                end
                
            elseif data.WL(jj-1)==3
                data.WSLScode(jj)=4; %loss+1 trial
               if data.bigSmall(jj) == 0.3 && data.ForceFree(jj)==0 %if data.bigSmall(jj)==0.3 && data.ForceFree(jj)==0
                   data.lose_shift(jj)=1;
               elseif data.omissionALL(jj)==1
                   data.lose_omit(jj)=1;
               end
            
            end
        end
        
        
    end
    
%Last row of data table is garbage, so delete this row. Double check & change this if the last row ends up being usable! 



% animalIDname = strsplit(filename);
% TotalWins = sum(data.WL(:)==1);
% TotalLosses = sum(data.WL(:)==3);
% TotalWinStay = sum(data.win_stay(:)==1);
% TotalLoseShift =sum(data.lose_shift(:)==1);
% WinStayPercent = TotalWinStay / TotalWins;
% LoseShiftPercent = TotalLoseShift / TotalLosses;

Descriptives = table;
Descriptives.TotalWins = sum(data.WL(:)==1);
Descriptives.TotalLosses = sum(data.WL(:)==3);
Descriptives.RiskPercent = (sum(data.bigRew(:)==1)/(sum(data.bigRew)+(sum(data.smallRew))))*100;
Descriptives.TotalWinStay = sum(data.win_stay(:)==1);
Descriptives.TotalLoseShift = sum(data.lose_shift(:)==1);
Descriptives.TotalLoseOmit = sum(data.lose_omit(:)==1);
Descriptives.WinStayPercent = Descriptives.TotalWinStay / Descriptives.TotalWins;
Descriptives.LoseShiftPercent = Descriptives.TotalLoseShift / Descriptives.TotalLosses;
Descriptives.LoseOmitPercent = Descriptives.TotalLoseOmit / Descriptives.TotalLosses;

% uncomment first one if you want to write descriptive statistics to file
%xlswrite(filename2,[TotalWins,TotalLosses,TotalWinStay,TotalLoseShift,WinStayPercent,LoseShiftPercent])
%xlswrite(filename,TotalWins,'TotalWins','B2')


end

end



