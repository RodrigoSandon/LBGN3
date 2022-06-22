import isx
import os, glob
from typing import List
import numpy as np
import scipy.signal as spsig
import matplotlib
import matplotlib.pyplot as plt

def find_paths_endswith(root_path, endswith) -> List:

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )

    return files

ROOT = r"/media/rory/Padlock_DT/BLA_Analysis/"
files = find_paths_endswith(ROOT, "cnmfe_cellset.isxd")
for f in files:
    print(f"CURR FILE: {f}")
    cell_set = isx.CellSet.read(f)
    dff_movie = isx.Movie.read(f.replace("cnmfe_cellset.isxd", "motion_corrected.isxd"))
    cell_set_out_path = f.replace("cnmfe_cellset.isxd", "cnmfe_cellset_accepts.isxd")
    if os.path.exists(cell_set_out_path):
        print("prev results removed")
        os.remove(cell_set_out_path)
    cell_set_out = isx.CellSet.write(cell_set_out_path, dff_movie.timing, cell_set.spacing)
    
    num_cells = cell_set.num_cells

    idx = 0
    for i in range(num_cells):
        cell_status = cell_set.get_cell_status(i)
        if str(cell_status) == "accepted":
            print(i)
            image = cell_set.get_cell_image_data(i)
            trace =  cell_set.get_cell_trace_data(i)
            idx_for_name = idx + 1
            if idx_for_name <= 9:
                name = f"C0{idx_for_name}"
            else:
                name = f"C{idx_for_name}"
            cell_set_out.set_cell_data(idx,image,trace,name)
            cell_set_out.set_cell_status(idx, "accepted")

            f,(orig_ax, new_ax) = plt.subplots(1,2)
            orig_ax.imshow(cell_set.get_cell_image_data(i))
            new_ax.imshow(cell_set_out.get_cell_image_data(i))
            plt.show()
            idx += 1
            
    break