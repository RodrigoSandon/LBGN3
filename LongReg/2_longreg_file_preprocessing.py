"""
Goals: 

-convert .mat file of cell registrations to csv
-also change the cell names of the file (.mat is based on index of the cells)
"""

from email.mime import base
import os, glob
import numpy as np
import pandas as pd
import h5py

# example .mat file path: /media/rory/Padlock_DT/BLA_Analysis/LongReg/Footprints/BLA-Insc-1/Results/cellRegistered_20220831_115812.mat
def find_paths(root_path, endswith: str):
    files = glob.glob(
        os.path.join(root_path, "**", endswith), recursive=True,
    )
    return files

def length_check(lst_1, lst_2):
    #print(len(lst_1), "vs", len(lst_2))
    if len(lst_1) == len(lst_2):
        print("lengths equal")
    else:
        print("lengths not equal")

def parse_footprint_name(mystr):
    cell = mystr.split("_")[0]
    return cell

def main():

    mat_path = "/media/rory/Padlock_DT/BLA_Analysis/LongReg/Footprints/BLA-Insc-1/Results/cellRegistered_20220831_115812.mat"
    footprints_dir_base_session = "/media/rory/Padlock_DT/BLA_Analysis/LongReg/Footprints/BLA-Insc-1/Pre-RDT RM"
    footprints_dir_session_1 = "/media/rory/Padlock_DT/BLA_Analysis/LongReg/Footprints/BLA-Insc-1/RDT D1"

    name_change_record_path = "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/Pre-RDT RM/naming_change_record.csv"


    f = h5py.File(mat_path,'r')
    # f structure: ['#refs#' 'cell_registered_struct']
    data = f.get("cell_registered_struct/cell_to_index_map")
    cell_to_index_map = np.array(data) # For converting to a NumPy array
    print(cell_to_index_map)

    #check the two arrays are of the same length
    ###### ADD MORE SESSIONS DEPEDING ON HOW MANY SESSIONS YOU'RE ANALYZING ######
    base_session = [int(i) for i in list(cell_to_index_map[0])]
    session_1 = [int(i) for i in list(cell_to_index_map[1])]

    length_check(base_session, session_1)
    # don't include cells in which their matches have a zero in it
    print(base_session)

    zipped = zip(base_session, session_1)

    idx_to_keep = [idx for idx, (x, y) in enumerate(zipped) if x != 0 and y != 0 ]
    print("idx_to_keep:", idx_to_keep) # idx starting at 0

    # now only keep cell index names of those that are not on the idx to omit list
    base_session_kept = []
    session_1_kept = []
    for idx in idx_to_keep:
        base_session_kept.append(base_session[idx])
        session_1_kept.append(session_1[idx])

    """print("CELLS KEPT")
    print(base_session_kept)
    print(session_1_kept)"""
    #Replace index names to cell names: index -> cellname as it was in .tif -> "new" cellname as in mouse's and session's "naming_change_record.csv"
    
    # list contents of footprints dir are same order as in the cell to index map
    """for i in os.listdir(footprints_dir_base_session):
        if ".mat" not in i:
            print(i)"""
    # so get the names based on the index name (-1 the index name provided in cellreg)
    old_names_base_session_kept = [parse_footprint_name(os.listdir(footprints_dir_base_session)[i-1]) for i in base_session_kept]
    old_names_session_1_kept = [parse_footprint_name(os.listdir(footprints_dir_session_1)[i-1]) for i in session_1_kept]

    print("CELLS REGISTERED")
    print(old_names_base_session_kept)
    print(old_names_session_1_kept)

    #open name change record path and change old names to new names
    df = pd.read_csv(name_change_record_path)

    new_names_base_session_kept = [df.index[df["Old Names"]==i] for i in old_names_base_session_kept]
    new_names_session_1_kept = [df.index[df["Old Names"]==i] for i in old_names_session_1_kept]

    # indexes for new names column
    print(new_names_base_session_kept)
    print(new_names_session_1_kept)

if __name__ == "__main__":
    main()