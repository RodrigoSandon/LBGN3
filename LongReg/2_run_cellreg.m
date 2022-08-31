
%% 1) Converting the Inscopix .tif cell spatial footprints to input format for CellReg

input_format='Inscopix'; %

% Choosing the files for conversion:
rootdir = "/media/rory/Padlock_DT/BLA_Analysis/LongReg/Footprints";

sessions = ["RM D1" "Pre-RDT RM" "RDT D1" "RDT D2" "RDT D3"];

mice = ["BLA-Insc-1" "BLA-Insc-2" "BLA-Insc-3" "BLA-Insc-5" "BLA-Insc-6" "BLA-Insc-7" "BLA-Insc-8" "BLA-Insc-9" "BLA-Insc-11" "BLA-Insc-13" "BLA-Insc-14" "BLA-Insc-15" "BLA-Insc-16" "BLA-Insc-18" "BLA-Insc-19"];

% /media/rory/Padlock_DT/BLA_Analysis/LongReg/Footprints/BLA-Insc-1/RDT D1/C00_footprint.tif
for i=1: length(mice)
    for j=1: length(sessions)
        footprint_dir = append(rootdir, "/",mice(i),"/", sessions(j));
        disp(footprint_dir)
        disp(class(footprint_dir))
        
        % check to see if session has any footprints
        footprint_dir_num_files = dir([footprint_dir '/*.tif'])
        print("here")
        
        if isequal(exist(footprint_dir), 7) && not(isequal(footprint_dir_num_files, 0))
            disp([mice(i) "and" sessions(j)]);
            footprints = dir(fullfile(footprint_dir, '*_footprint.tif'));
            
            this_session_num_cells = length(footprints);

            % get size of footprints first
            fname = footprints(1);
            fname = fullfile(fname.folder, fname.name)
            footprint = read(Tiff(fname,'r'));  

            %
            this_session_converted_footprints=zeros(this_session_num_cells,size(footprint, 1), size(footprint, 2));

            %
            for n=1:this_session_num_cells
                disp(n)
                fname = footprints(n);
                fname = fullfile(fname.folder, fname.name);

                %
                footprint = read(Tiff(fname,'r'));    
                this_session_converted_footprints(n,:,:)= footprint;

            end
            fname_out = fullfile(footprints(1).folder, append('spatial_footprints_', mice(i), sessions(j), '.mat'));
            save(fname_out,'this_session_converted_footprints','-v7.3')
        end
    end
end


%% 2) Running CellReg

% 2a) Setting paths for the cell registration procedure

% 2b) Stage 1 - Loading the spatial footprints of cellular activity:

% 2c) Stage 2 - Aligning all the sessions to a reference coordinate system:

% 2d) Stage 3 (part a) - Calculating the similarities distributions from the data:

% 2e) Stage 3 (part b) - Compute a probabilistic model:

% 2f) Stage 4 - Initial cell registration

% 2g) Stage 5 - Final cell registration: