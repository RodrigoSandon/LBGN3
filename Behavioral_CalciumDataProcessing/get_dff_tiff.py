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
    root_path = "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/"
    cellsets, roots = get_input_cell_set_files(root_path)  # should have the same index

    for i in range(len(cellsets)):
        get_dff_and_tiff(
            cellsets[i],
            os.path.join(roots[i], "dff_traces.csv"),
            os.path.join(roots[i], "cell_"),
            "start",
            "",
        )

def main2():

    ROOT = r"/media/rory/Padlock_DT/BLA_Analysis/"

    mice = [
            "BLA-Insc-14",
            "BLA-Insc-15",
            "BLA-Insc-16"]

    sessions = ["Pre-RDT RM", "RDT D1"]

    for mouse in mice:
        print("CURR MOUSE", mouse)
        ptp = ptp_autoencoder(mouse.lower())

        for session in sessions:

            session_dir = f"{ROOT}/{ptp}/{mouse}/{session}"

            intermediate_base = possible_intermediate(ptp, session_dir)

            raw_dff_traces = f"{ROOT}/{ptp}/{mouse}/{session}/{intermediate_base}dff_traces.csv"

            isx.export_cell_set_to_csv_tiff(
                f"{ROOT}/{ptp}/{mouse}/{session}/{intermediate_base}cnmfe_cellset.isxd",
                f"{ROOT}/{ptp}/{mouse}/{session}/{intermediate_base}dff_traces.csv",
                f"{ROOT}/{ptp}/{mouse}/{session}/{intermediate_base}",
                "start",
                "",
            )


#main()
main2()
