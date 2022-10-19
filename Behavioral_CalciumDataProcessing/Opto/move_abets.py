from genericpath import isdir
import pandas as pd
import numpy as np
import shutil
import os, glob
from pathlib import Path
from typing import List
from BehavioralSession_opto import BehavioralSession
from BehavioralUtilities_opto import BehavioralUtilities


def find_paths_startswith_and_endswith(root_path, startswith, endswith) -> List:

    files = glob.glob(
        os.path.join(root_path, "**", "%s*%s") % (startswith, endswith),
        recursive=True,
    )

    return files

def parse_abet_file(abet_file_path):

    mouse_date = abet_file_path.split("/")[-1]
    mouse = mouse_date.split(" ")[0]
    date = mouse_date.split(" ")[1].replace(".csv","")

    return mouse, date

def find_path_endswith(session_path, endswith):
    files = glob.glob(
        os.path.join(session_path, "**", "*%s") % (endswith),
        recursive=True,
    )
    return files

def main():
    # move abets and movies?
    
    ROOT = r"/media/rory/RDT VIDS/ABET_files_opto/"
    DST_1 = r"/media/rory/RDT VIDS/BORIS_merge"
    DST_2 = r"/media/rory/RDT VIDS/BORIS"

    files = find_paths_startswith_and_endswith(ROOT, "RRD", ".csv")

    for count, f in enumerate(files):
        print(f"file {count}/{len(files)}")
        print("CURRENT PATH: ", f)

        try:
            file_name = f.split("/")[-1]

            mouse, date = parse_abet_file(f)

            movi = find_paths_startswith_and_endswith(DST_1, mouse, "_merged_resized_grayscaled.mp4")
            dst_root_dir = f"{DST_1}/{mouse}/"
            if len(movi) == 0:
                movi = find_paths_startswith_and_endswith(DST_2, mouse, "_merged_resized_grayscaled.mp4")
                dst_root_dir = f"{DST_2}/{mouse}/"
                if len(movi) == 0:
                    print(f"no found vid for {mouse}")
                    pass
            print(mouse)

            # check if mouse exists in either boris
            if os.path.isdir(f"{DST_1}/{mouse}"):
                dst_abet_dir = f"{DST_1}/{mouse}/{file_name}"

            elif os.path.isdir(f"{DST_2}/{mouse}"):
                dst_abet_dir = f"{DST_2}/{mouse}/{file_name}"

            # check if movi exists in folder
            
            if len(movi) != 0:
                #MOVI CAN BE FROM BORIS OR MORIS_MERGE
                movi_name = movi[0].split("/")[-1]

                # check if movi in mouse name already
                movi_found = find_path_endswith(dst_root_dir, movi_name)
                if len(movi_found) == 0:
                    # this means movi not in mouse folder
                    shutil.move(movi[0], os.path.join(dst_root_dir, movi_name))
                

            shutil.copy(f, dst_abet_dir)
        except (FileNotFoundError) as e:
            print(e)
            pass

main()