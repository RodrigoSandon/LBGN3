%%
load('FearCond_data_02152023.mat')

animalIDs = string(fieldnames(data.streams));
treatment = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2];

ChrimsonR_ind = treatment == 1';
mCherry_ind = treatment ~=(ChrimsonR_ind')';




%% Conditioning

clear CS_*


num_columns = 3; 

CS_Conditioning = cell(1, num_columns);
for col = 1:num_columns
    
    for animal = 1:size(animalIDs, 1)
        currentAnimal = animalIDs(animal);

            column_data = data.streams.(currentAnimal).Conditioning.Freezing{col}(:,1);

            CS_Conditioning{col}(:,animal) = column_data;

    end



end

for ii = 1:size(CS_Conditioning, 2)
    CS_ChrimsonR{ii}(:,:) = CS_Conditioning{1,ii}(:,treatment==1);
    CS_ChrimsonR_mouse_mean(ii,:) = mean(CS_ChrimsonR{ii});
    CS_ChrimsonR_mouse_SEM(ii,:) = sem(CS_ChrimsonR{ii});
    CS_mCherry{ii}(:,:) = CS_Conditioning{1,ii}(:,treatment==2);
    CS_mCherry_mouse_mean(ii, :) = mean(CS_mCherry{ii});
    CS_mCherry_mouse_SEM(ii,:) = sem(CS_mCherry{ii});

end
CS_mCherry_group_mean = mean(CS_mCherry_mouse_mean,2);
CS_mCherry_mouse_mean_rows = CS_mCherry_mouse_mean';
CS_mCherry_group_SEM = (std(CS_mCherry_mouse_mean_rows,0)/sqrt(size(CS_mCherry_mouse_mean_rows, 1)))';

CS_ChrimsonR_group_mean = mean(CS_ChrimsonR_mouse_mean,2);
CS_ChrimsonR_mouse_mean_rows = CS_ChrimsonR_mouse_mean';
CS_ChrimsonR_group_SEM = (std(CS_ChrimsonR_mouse_mean_rows,0)/sqrt(size(CS_ChrimsonR_mouse_mean_rows, 1)))';


CS_bins = 1:5:size(CS_Conditioning, 2);

CS_num = 1:num_columns;


figure;
errorbar(CS_num,CS_ChrimsonR_group_mean,CS_ChrimsonR_group_SEM,"-s","MarkerSize",10,...
    "Color", [1 0.5 0],"MarkerEdgeColor",[1 0.5 0],"MarkerFaceColor",[1 0.5 0])
hold on;
errorbar(CS_num,CS_mCherry_group_mean,CS_mCherry_group_SEM,"-o","MarkerSize",10,...
    "Color", "black", "MarkerEdgeColor","black","MarkerFaceColor","black")
    set(gca, 'XTick', CS_num)
    set(gca, 'XLim', [0 4], 'YLim', [0 50])
    set(gcf,'Position',[100 100 500 500])



%% Extinction

clear CS_*

num_columns = 50; 

CS_Extinction = cell(1, num_columns);
for col = 1:num_columns
    
    for animal = 1:size(animalIDs, 1)
        currentAnimal = animalIDs(animal);

            column_data = data.streams.(currentAnimal).Extinction.Freezing{col}(:,1);

            CS_Extinction{col}(:,animal) = column_data;

    end



end


for ii = 1:size(CS_Extinction, 2)
    CS_ChrimsonR{ii}(:,:) = CS_Extinction{1,ii}(:,treatment==1);
    CS_ChrimsonR_mouse_mean(ii,:) = mean(CS_ChrimsonR{ii});
    CS_ChrimsonR_mouse_SEM(ii,:) = sem(CS_ChrimsonR{ii});
    CS_mCherry{ii}(:,:) = CS_Extinction{1,ii}(:,treatment==2);
    CS_mCherry_mouse_mean(ii, :) = mean(CS_mCherry{ii});
    CS_mCherry_mouse_SEM(ii,:) = sem(CS_mCherry{ii});

end

CS_mCherry_group_mean = mean(CS_mCherry_mouse_mean,2);
CS_mCherry_mouse_mean_rows = CS_mCherry_mouse_mean';
CS_mCherry_group_SEM = (std(CS_mCherry_mouse_mean_rows,0)/sqrt(size(CS_mCherry_mouse_mean_rows, 1)))';

CS_ChrimsonR_group_mean = mean(CS_ChrimsonR_mouse_mean,2);
CS_ChrimsonR_mouse_mean_rows = CS_ChrimsonR_mouse_mean';
CS_ChrimsonR_group_SEM = (std(CS_ChrimsonR_mouse_mean_rows,0)/sqrt(size(CS_ChrimsonR_mouse_mean_rows, 1)))';


CS_bins = 1:5:size(CS_Extinction, 2);

CS_num = 1:num_columns;


figure;
errorbar(CS_num,CS_ChrimsonR_group_mean,CS_ChrimsonR_group_SEM,"-s","MarkerSize",10,...
    "Color", [1 0.5 0],"MarkerEdgeColor",[1 0.5 0],"MarkerFaceColor",[1 0.5 0])
hold on;
errorbar(CS_num,CS_mCherry_group_mean,CS_mCherry_group_SEM,"-o","MarkerSize",10,...
    "Color", "black", "MarkerEdgeColor","black","MarkerFaceColor","black")
%     set(gca, 'XTick', CS_num(0:2:50))
    set(gca, 'XLim', [0 CS_num(end)], 'YLim', [0 50])
    set(gcf,'Position',[100 100 1000 500])

%% Retrieval

clear CS_*

num_columns = 5; 

CS_Retrieval = cell(1, num_columns);
for col = 1:num_columns
    
    for animal = 1:size(animalIDs, 1)
        currentAnimal = animalIDs(animal);

            column_data = data.streams.(currentAnimal).Retrieval.Freezing{col}(:,1);

            CS_Retrieval{col}(:,animal) = column_data;

    end



end


for ii = 1:size(CS_Retrieval, 2)
    CS_ChrimsonR{ii}(:,:) = CS_Retrieval{1,ii}(:,treatment==1);
    CS_ChrimsonR_mouse_mean(ii,:) = mean(CS_ChrimsonR{ii});
    CS_ChrimsonR_mouse_SEM(ii,:) = sem(CS_ChrimsonR{ii});
    CS_mCherry{ii}(:,:) = CS_Retrieval{1,ii}(:,treatment==2);
    CS_mCherry_mouse_mean(ii, :) = mean(CS_mCherry{ii});
    CS_mCherry_mouse_SEM(ii,:) = sem(CS_mCherry{ii});

end

CS_mCherry_group_mean = mean(CS_mCherry_mouse_mean,2);
CS_mCherry_mouse_mean_rows = CS_mCherry_mouse_mean';
CS_mCherry_group_SEM = (std(CS_mCherry_mouse_mean_rows,0)/sqrt(size(CS_mCherry_mouse_mean_rows, 1)))';

CS_ChrimsonR_group_mean = mean(CS_ChrimsonR_mouse_mean,2);
CS_ChrimsonR_mouse_mean_rows = CS_ChrimsonR_mouse_mean';
CS_ChrimsonR_group_SEM = (std(CS_ChrimsonR_mouse_mean_rows,0)/sqrt(size(CS_ChrimsonR_mouse_mean_rows, 1)))';


CS_bins = 1:5:size(CS_Retrieval, 2);

CS_num = 1:num_columns;

figure;
errorbar(CS_num,CS_ChrimsonR_group_mean,CS_ChrimsonR_group_SEM,"-s","MarkerSize",10,...
    "Color", [1 0.5 0],"MarkerEdgeColor",[1 0.5 0],"MarkerFaceColor",[1 0.5 0])
hold on;
errorbar(CS_num,CS_mCherry_group_mean,CS_mCherry_group_SEM,"-o","MarkerSize",10,...
    "Color", "black", "MarkerEdgeColor","black","MarkerFaceColor","black")
    set(gca, 'XTick', CS_num)
    set(gca, 'XLim', [0 6], 'YLim', [0 50])
    set(gcf,'Position',[100 100 700 500])