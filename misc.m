
rootdir = "/media/rory/Padlock_DT/BLA_Analysis/LongReg/Footprints";

sessions = ["RM D1" "Pre-RDT RM" "RDT D1" "RDT D2" "RDT D3"];

mice = ["BLA-Insc-1" "BLA-Insc-2" "BLA-Insc-3" "BLA-Insc-5" "BLA-Insc-6" "BLA-Insc-7" "BLA-Insc-8" "BLA-Insc-9" "BLA-Insc-11" "BLA-Insc-13" "BLA-Insc-14" "BLA-Insc-15" "BLA-Insc-16" "BLA-Insc-18" "BLA-Insc-19"];

% /media/rory/Padlock_DT/BLA_Analysis/LongReg/Footprints/BLA-Insc-1/RDT D1/C00_footprint.tif
for i=1: length(mice)
    for j=1: length(sessions)
        footprint_dir = append(rootdir, "/",mice(i),"/", sessions(j));
        if isequal(exist(footprint_dir), 7)
            disp([mice(i) "and" sessions(j)]);
            footprints = dir(fullfile(footprint_dir, '*_footprint.tif'));
            disp("footprints found!")
        end
    end
end
        
