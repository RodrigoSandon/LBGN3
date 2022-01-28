import isx
from pathlib import Path
from typing import List
import os

# example path: /media/rory/RDT VIDS/PTP Inscopix #1/Session .../
def get_dff_and_tiff(
    input_cell_set_files, output_csv_file, output_tiff_file, time_ref, output_props_file
):
    isx.export_cell_set_to_csv_tiff(
        input_cell_set_files,
        output_csv_file,
        output_tiff_file,
        time_ref,
        output_props_file,
    )


# /media/rory/RDT VIDS/PTP_Inscopix_#1/BLA-Insc-1/Session-20210118/cnmfe_cellset.isxd
def get_input_cell_set_files(root_path: Path):
    cell_set_files = []
    root_paths_to_cell_set_files = []

    for root, dirs, files in os.walk(root_path):
        for name in files:

            if name.startswith("cnmfe_cellset.isxd"):
                path_to_cellset = os.path.join(root, name)
                cell_set_files.append(path_to_cellset)
                root_paths_to_cell_set_files.append(root)
    # now have all paths to the cellset isxds
    print(*cell_set_files, sep="\n")
    print("Number of cell sets: %s" % (len(cell_set_files)))
    return cell_set_files, root_paths_to_cell_set_files


def main():
    root_path = "/media/rory/RDT VIDS/PTP_Inscopix_#1/"
    cellsets, roots = get_input_cell_set_files(root_path)  # should have the same index

    for i in range(len(cellsets)):
        get_dff_and_tiff(
            cellsets[i],
            os.path.join(roots[i], "dff_traces.csv"),
            os.path.join(roots[i], "cell_"),
            "start",
            "",
        )


main()
