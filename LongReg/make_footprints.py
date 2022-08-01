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


mice = [
    "BLA-Insc-1",
    "BLA-Insc-6",
    "BLA-Insc-15"
]

ptp = {
    "BLA-Insc-1": "PTP_Inscopix_#1",
    "BLA-Insc-6": "PTP_Inscopix_#3",
    "BLA-Insc-15": "PTP_Inscopix_#5"
}

sessions = [
    "Pre-RDT RM",
    "RDT D1",
    "RDT D2",
    "RDT D3"
]

for mouse in mice:
    for session in sessions:
        print(mouse, session)
        try:
            src = find_paths(f'/media/rory/Padlock_DT/BLA_Analysis/{ptp[mouse]}/{mouse}/{session}', 'motion_corrected.isxd')[0]
            dst = src.replace(".isxd", ".tif")
            isx.export_movie_to_tiff(src, dst)
        
            footprints, traces = inscopix_cnmfe.run_cnmfe(
                input_movie_path=dst, 
                output_dir_path='output', 
                output_filetype=0,
                average_cell_diameter=7,
                min_pixel_correlation=0.8,
                min_peak_to_noise_ratio=10.0,
                gaussian_kernel_size=0,
                closing_kernel_size=0,
                background_downsampling_factor=2,
                ring_size_factor=1.4,
                merge_threshold=0.7,
                num_threads=4,
                processing_mode=2,
                patch_size=80,
                patch_overlap=20,
                output_units=1,
                deconvolve=0,
                verbose=1
            )

            dst_dir = f"/media/rory/Padlock_DT/BLA_Analysis/LongReg/CellReg/{mouse}/{session}"
            os.makedirs(dst_dir, exist_ok=True)
            #print(len(footprints))

            for footprint_idx in range(0,len(footprints)):
                img = Image.fromarray(footprints[footprint_idx])
                img.save(f"{dst_dir}/cell_{footprint_idx}.tif")

        except Exception as e:
            print(e)
            pass

        