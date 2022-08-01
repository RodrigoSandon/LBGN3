%% Converting the Inscopix .tif cell spatial footprints to input format for CellReg

input_format='Inscopix'; %

%% Choosing the files for conversion:
[files_path]=uigetdir('Choose the location of the footprints: ' );
file_names = dir(fullfile(files_path, 'cell_*.tif'));
%print(file_names)
%
this_session_num_cells = length(file_names);

% get size of footprints first
for i=1: length(file_names)
    fname = file_names(i);
    fname = fullfile(fname.folder, fname.name)
    footprint = read(Tiff(fname,'r'));  

    % I have a feeling the sizes have to be all the footprints combined
    this_session_converted_footprints=zeros(this_session_num_cells, size(footprint, 1), size(footprint, 2))
   

    %
    
    this_session_converted_footprints(i,:,:)= footprint;
    
    fname_out = fullfile(file_names(i).folder, 'converted_all_footprints.mat');
    save(fname_out,'this_session_converted_footprints','-v7.3')
end
