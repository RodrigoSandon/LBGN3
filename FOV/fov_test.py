import isx

statuses = [ 'accepted',	 'accepted',	 'rejected',	 'rejected',	 'accepted',	 'accepted',	 'rejected',	 'accepted',	 'rejected',	 'rejected',	 'rejected',	 'rejected',	 'rejected',	 'accepted',	 'rejected',	 'accepted'	, 'rejected',	 'accepted',	 'accepted'	, 'accepted',	 'accepted'	, 'rejected',	 'accepted',	 'accepted',	 'rejected'	, 'rejected',	 'rejected'	 ,'rejected',	 'rejected',	 'accepted'	 ,'rejected'	, 'rejected',	 'rejected',	 'rejected'	, 'rejected',	 'rejected'	 ,'rejected'	, 'rejected',	 'accepted'	, 'rejected'	, 'accepted'	, 'rejected'	, 'rejected'	, 'accepted'	, 'rejected',	 'accepted'	, 'rejected'	, 'rejected'	, 'rejected'	, 'rejected'	, 'accepted',	 'rejected',	 'rejected'	, 'rejected'	, 'accepted'	, 'rejected',	 'accepted'	, 'rejected',	 'rejected']

#isx.create_cell_map("/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/cnmfe_cellset.isxd",selected_cell_statuses=statuses,output_isxd_cell_map_file="/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/cellmap.isxd",output_tiff_cell_map_file="/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/cellmap.tiff")
dff_movie = isx.Movie.read("/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/motion_corrected.isxd")

cell_set = isx.CellSet.read("/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/cnmfe_cellset.isxd")
cell_set_out = isx.CellSet.write("/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/cnmfe_cellset_accepts.isxd", dff_movie.timing, cell_set.spacing)
num_cells = cell_set.num_cells

idx = 0
for i in range(num_cells):
    cell_status = cell_set.get_cell_status(i)
    if str(cell_status) == "accepted":
        print(cell_status)
        image = cell_set.get_cell_image_data(i)
        trace =  cell_set.get_cell_trace_data(i)
        idx_for_name = idx + 1
        if idx_for_name <= 9:
            name = f"C0{idx_for_name}"
        else:
            name = f"C{idx_for_name}"
        cell_set_out.set_cell_data(idx,image,trace,name)
        idx += 1