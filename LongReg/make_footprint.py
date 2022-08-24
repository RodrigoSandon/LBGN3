import inscopix_cnmfe
import isx
import os, glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm

def find_paths(root_path, endswith: str):
    files = glob.glob(
        os.path.join(root_path, "**", endswith), recursive=True,
    )
    return files

src = "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/motion_corrected.isxd"
cellset_path = "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/cnmfe_cellset_accepts.isxd"
cnmfe_cellset = isx.CellSet.read(cellset_path)
num_cells = cnmfe_cellset.num_cells
mouse = cnmfe_cellset.split("/")[6]
session = cnmfe_cellset.split("/")[7]

print(num_cells)


dst = src.replace(".isxd", ".tif")
try:
    isx.export_movie_to_tiff(src, dst)
except Exception as e:
    print(e)
    pass

footprints, traces = inscopix_cnmfe.run_cnmfe(
    input_movie_path=dst, 
    output_dir_path='output', 
    output_filetype=0,
    average_cell_diameter=16,
    min_pixel_correlation=0.7,
    min_peak_to_noise_ratio=8,
    gaussian_kernel_size=4,
    closing_kernel_size=0,
    background_downsampling_factor=1,
    gSig= 4,
    ring_size_factor=1.125,
    merge_threshold=0.3,
    num_threads=5,
    processing_mode="parallel_patches",
    patch_size=80,
    patch_overlap=20,
    output_units="df_over_noise",
)

for i in range(num_cells):
    cell_status = cnmfe_cellset.get_cell_status(i)
    cell_name = cnmfe_cellset.get_cell_name(i)
    if str(cell_status) == "accepted":
        trace_accepted =  cnmfe_cellset.get_cell_trace_data(i)
        trace_accepted_sub = trace_accepted[:10]

        # match this with the traces of the footprints
        for trace_idx in range(0, len(traces)):
            footprint_trace_sub = traces[trace_idx][:10]
            if trace_accepted_sub == footprint_trace_sub:
                print(f"footprint {trace_idx} matches accepted cell {cell_name}")
                print(footprint_trace_sub)
                print(trace_accepted_sub)

dst_dir = f"/media/rory/Padlock_DT/BLA_Analysis/LongReg/CellReg/{mouse}/{session}"
os.makedirs(dst_dir, exist_ok=True)
#print(len(footprints))

for footprint_idx in range(0,len(footprints)):
    img = Image.fromarray(footprints[footprint_idx])
    img.save(f"{dst_dir}/cell_{footprint_idx}.tif")
    

