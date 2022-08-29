import inscopix_cnmfe
import isx
import os, glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

def find_paths(root_path, endswith: str):
    files = glob.glob(
        os.path.join(root_path, "**", endswith), recursive=True,
    )
    return files

def ptp_autoencoder(mouse_tolower: str) -> str:
    
    mouse_tolower = mouse_tolower.lower()
    d = {
        "PTP_Inscopix_#1": ["bla-insc-1", "bla-insc-2", "bla-insc-3"],
        "PTP_Inscopix_#3": ["bla-insc-5", "bla-insc-6", "bla-insc-7"],
        "PTP_Inscopix_#4": ["bla-insc-8", "bla-insc-9", "bla-insc-11", "bla-insc-13"],
        "PTP_Inscopix_#5": ["bla-insc-14", "bla-insc-15", "bla-insc-16", "bla-insc-18", "bla-insc-19"]
    }

    for key in d.keys():
        if mouse_tolower in d[key]:
            return key

def possible_intermediate(ptp, session_dir):
    # there is an intermediate
    res = ""
    if "PTP_Inscopix_#1" != ptp:
        for dir in os.listdir(session_dir):
            if "BLA" in dir:
                res = dir + "/"
    return res
            

def main():

    ROOT = r"/media/rory/Padlock_DT/BLA_Analysis"

    session_types = [
            "RM D1",        
            "Pre-RDT RM",
            "RDT D1",
            "RDT D2",
            'RDT D3'
        ]
    
    mice = ["BLA-Insc-1",
            "BLA-Insc-2",
            "BLA-Insc-3",
            "BLA-Insc-5",
            "BLA-Insc-6",
            "BLA-Insc-7",
            "BLA-Insc-8",
            "BLA-Insc-9",
            "BLA-Insc-11",
            "BLA-Insc-13",
            "BLA-Insc-14",
            "BLA-Insc-15",
            "BLA-Insc-16",
            "BLA-Insc-18",
            "BLA-Insc-19",]

    # /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/Pre-RDT RM/motion_corrected.tif
    
    for mouse in mice:
        for session in session_types:
            try:
                ptp = ptp_autoencoder(mouse.lower())
                session_dir = f"{ROOT}/{ptp}/{mouse}/{session}"
                intermediate = possible_intermediate(ptp, session_dir)

                motion_corrected_path_isxd = f"{session_dir}/{intermediate}motion_corrected.isxd"
                motion_corrected_dir = f"{session_dir}/{intermediate}"

                footprint_results_dir = f"/media/rory/Padlock_DT/BLA_Analysis/LongReg/Footprints/{mouse}/{session}"
                cellset_path = motion_corrected_path_isxd.replace("motion_corrected.isxd","cnmfe_cellset.isxd")

                motion_corrected_path_tif = motion_corrected_path_isxd.replace(".isxd", ".tif")

                # if the sesssion doesn't include a motion corrected tif, then run the convert to tif command
            
                run_convert_to_tif = True
                for dir in os.listdir(motion_corrected_dir):
                    if "motion_corrected.tif" == dir:
                        run_convert_to_tif = False


                if run_convert_to_tif == True:
                    isx.export_movie_to_tiff(motion_corrected_path_isxd, motion_corrected_path_tif)

                if os.path.isdir(footprint_results_dir) == False:
                    # footprints results dir is empty
                    print(f"WORKING ON: {mouse} {session}")
                    footprints, traces = inscopix_cnmfe.run_cnmfe(
                        input_movie_path=motion_corrected_path_tif, 
                        output_dir_path='output', 
                        output_filetype=0,
                        average_cell_diameter=16,
                        min_pixel_correlation=0.7,
                        min_peak_to_noise_ratio=8,
                        gaussian_kernel_size=4,
                        closing_kernel_size=0,
                        background_downsampling_factor=1,
                        ring_size_factor=1.125,
                        merge_threshold=0.3,
                        num_threads=5,
                        processing_mode=2,
                        patch_size=80,
                        patch_overlap=20,
                        output_units=1,
                    )
                
                    cnmfe_cellset = isx.CellSet.read(cellset_path)
                    num_cells = cnmfe_cellset.num_cells

                    os.makedirs(footprint_results_dir, exist_ok=True)

                    num_matches = 0
                    # iterating over accepted cells
                    for i in range(num_cells):
                        cell_status = cnmfe_cellset.get_cell_status(i)
                        cell_name = cnmfe_cellset.get_cell_name(i)
                        if str(cell_status) == "accepted":
                            cell_trace = list(cnmfe_cellset.get_cell_trace_data(i))
                            cell_trace.sort()

                            for trace_idx, footprint_trace in enumerate(traces):
                                footprint_trace = list(footprint_trace)
                                footprint_trace.sort()

                                if cell_trace == footprint_trace:
                                    footprint_idx = trace_idx
                                    #print(f"footprint {footprint_idx} matches accepted cell {cell_name}")
                                    img = Image.fromarray(footprints[footprint_idx])
                                    img.save(f"{footprint_results_dir}/{cell_name}_footprint.tif")
                                    num_matches += 1
                                    break
                                    
                    print(num_matches)

            except Exception as e:
                print(e)
                pass
                    

if __name__ == "__main__":
    main()