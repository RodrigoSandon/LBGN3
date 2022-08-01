%% Converting the Inscopix .tif cell spatial footprints to input format for CellReg

input_format='Inscopix'; %

%% Choosing the files for conversion:

[files_path]=uigetdir('Choose the location of the footprints: ' );
file_names = dir(fullfile(files_path, '**', 'cell_*.tif'));

%
this_session_num_cells = length(file_names);

% get size of footprints first
fname = file_names(1);
fname = fullfile(fname.folder, fname.name)
footprint = read(Tiff(fname,'r'));  

%
this_session_converted_footprints=zeros(this_session_num_cells,size(footprint, 1), size(footprint, 2));

%
for n=1:this_session_num_cells
    disp(n)
    fname = file_names(n);
    fname = fullfile(fname.folder, fname.name);
    
    %
    footprint = read(Tiff(fname,'r'));    
    this_session_converted_footprints(n,:,:)= footprint;
    
end
fname_out = fullfile(file_names(1).folder, 'spatial_footprints.mat');
save(fname_out,'this_session_converted_footprints','-v7.3')
