import isx

statuses = [ 'accepted',	 'accepted',	 'rejected',	 'rejected',	 'accepted',	 'accepted',	 'rejected',	 'accepted',	 'rejected',	 'rejected',	 'rejected',	 'rejected',	 'rejected',	 'accepted',	 'rejected',	 'accepted'	, 'rejected',	 'accepted',	 'accepted'	, 'accepted',	 'accepted'	, 'rejected',	 'accepted',	 'accepted',	 'rejected'	, 'rejected',	 'rejected'	 ,'rejected',	 'rejected',	 'accepted'	 ,'rejected'	, 'rejected',	 'rejected',	 'rejected'	, 'rejected',	 'rejected'	 ,'rejected'	, 'rejected',	 'accepted'	, 'rejected'	, 'accepted'	, 'rejected'	, 'rejected'	, 'accepted'	, 'rejected',	 'accepted'	, 'rejected'	, 'rejected'	, 'rejected'	, 'rejected'	, 'accepted',	 'rejected',	 'rejected'	, 'rejected'	, 'accepted'	, 'rejected',	 'accepted'	, 'rejected',	 'rejected']

isx.create_cell_map("/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/cnmfe_cellset.isxd", rgb="red",selected_cell_statuses=statuses,output_isxd_cell_map_file="/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/cellmap.isxd",output_tiff_cell_map_file="/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/cellmap.tiff")

