%% Navigate to folder where SLEAP data are (should be structured as a folder with the animal ID, then subfolders for each session)
% Within each subfolder should be a .slp file, a .h5 file, and folders
% containing each of the body points tracked by SLEAP. At the time of
% writing (12/1/2022), we are tracking 6 body points




% get_dir = dir('h:\MATLAB\TDTbin2mat\SLEAP Data_processed\67');
get_dir = dir(pwd); %get top level directory
dir_string = get_dir.folder; %turn directory folder into a string 
dir_string_split = split(dir_string,"\"); %split the string at the \ delimiter (this will be used to get the animal ID, as final part of the string should be the animal ID)
x_motion = table;
y_motion = table;
%MATLAB dir generates 2 nonsense variables at the top of every directory,
%skip these (so start at 3)
for i = 3:size(get_dir,1)
    folder_name = dir_string+"\"+get_dir(i).name; %get the first folder name by combining the directory w/ the folder name
    cd(folder_name); 
    sleap_dir = dir(folder_name); %get files in the folder
    slp_fname = dir('*.slp'); %identify the .slp file name, which will be used to get the session date for file naming later
    slp_fname_split = split(slp_fname.name,"."); %so that we can get the date later
    
    for q = 5:size(sleap_dir,1)
        sleap_dir_string = sleap_dir.folder;
        sleap_folder_name = sleap_dir_string+"\"+sleap_dir(q).name;
        sleap_folder_name_strings = split(sleap_folder_name, "\");
        cd(sleap_folder_name);
        xlfiles = dir('*.csv');
        xls_fname = xlfiles.name ;   % file name
        xls_body_type = sleap_folder_name_strings(end);
%         date = char(slp_fname_split(2));
        
        session = sleap_folder_name_strings(end-1);
        session = regexprep(session, ' ', '_');
        animal_ID = char(dir_string_split(end));
        new_fname = char(strcat(animal_ID,'_', session,'_',xls_body_type,'_sleap_data','.csv'));
        raw_SLEAP_table = readtable(xls_fname);
        x_motion.idx_time = raw_SLEAP_table.idx_time;
        x_motion.(xls_body_type) = raw_SLEAP_table.x_pix;
        y_motion.idx_time = raw_SLEAP_table.idx_time;
        y_motion.(xls_body_type) = raw_SLEAP_table.y_pix;
        writetable(raw_SLEAP_table,new_fname)
    end
    cd(folder_name)
    
    writetable(x_motion, strcat(animal_ID,'_',session,'_','x_motion.csv'))
    writetable(y_motion, strcat(animal_ID,'_',session,'_','y_motion.csv'))
    

end