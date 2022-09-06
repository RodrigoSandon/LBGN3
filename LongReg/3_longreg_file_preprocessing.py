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
def find_paths_endswith(root_path, startwith: str):
    files = glob.glob(
        os.path.join(root_path, "**", f"{startwith}*"), recursive=True,
    )

    if len(files) == 1:
        files = files[0]

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

    mice = ["BLA-Insc-1",
            "BLA-Insc-6",
            "BLA-Insc-13",
            "BLA-Insc-14",
            "BLA-Insc-15",
            "BLA-Insc-16"]

    sessions = ["Pre-RDT RM", "RDT D1"]

    for mouse in mice:
        print("CURR MOUSE", mouse)
        ptp = ptp_autoencoder(mouse.lower())
        mouse_dir = f"{ROOT}/{ptp}/{mouse}"
        session_dir_base = f"{ROOT}/{ptp}/{mouse}/{sessions[0]}"
        session_dir_1 = f"{ROOT}/{ptp}/{mouse}/{sessions[1]}"

        intermediate_base = possible_intermediate(ptp, session_dir_base)
        intermediate_1 = possible_intermediate(ptp, session_dir_1)
    
        mat_path = find_paths_endswith(f"{ROOT}/LongReg/Footprints/{mouse}/Results/", "cellRegistered")
        print(mat_path)
        footprints_dir_base_session = f"{ROOT}/LongReg/Footprints/{mouse}/{sessions[0]}"
        footprints_dir_session_1 = f"{ROOT}/LongReg/Footprints/{mouse}/{sessions[1]}"

        name_change_record_path_base_session = f"{ROOT}/{ptp}/{mouse}/{sessions[0]}/{intermediate_base}naming_change_record.csv"
        name_change_record_path_session_1 = f"{ROOT}/{ptp}/{mouse}/{sessions[1]}/{intermediate_1}naming_change_record.csv"

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

        zipped = zip(base_session, session_1)

        idx_to_keep = [idx for idx, (x, y) in enumerate(zipped) if x != 0 and y != 0 ]
        print("idx_to_keep:", idx_to_keep) # idx starting at 0

        # now only keep cell index names of those that are not on the idx to omit list
        base_session_kept = []
        session_1_kept = []
        for idx in idx_to_keep:
            base_session_kept.append(base_session[idx])
            session_1_kept.append(session_1[idx])

        #Replace index names to cell names: index -> cellname as it was in .tif -> "new" cellname as in mouse's and session's "naming_change_record.csv"

        # list contents of footprints dir are same order as in the cell to index map
        # so get the names based on the index name (-1 the index name provided in cellreg)
        old_names_base_session_kept = [parse_footprint_name(os.listdir(footprints_dir_base_session)[i-1]) for i in base_session_kept]
        old_names_session_1_kept = [parse_footprint_name(os.listdir(footprints_dir_session_1)[i-1]) for i in session_1_kept]

        #open name change record path and change old names to new names
        name_change_record_base_session_df = pd.read_csv(name_change_record_path_base_session)
        name_change_record_session_1_df = pd.read_csv(name_change_record_path_session_1)
        # for some reason there are weird spaces in the df
                
        name_change_record_base_session_df["Old Names"] = [i.replace(" ", "") for i in list(name_change_record_base_session_df["Old Names"]) if " " in i]
        name_change_record_session_1_df["Old Names"] = [i.replace(" ", "") for i in list(name_change_record_session_1_df["Old Names"]) if " " in i]

        # get row index for new names column
        new_names_base_session_kept = [name_change_record_base_session_df.loc[int(name_change_record_base_session_df.set_index('Old Names').index.get_loc(i)), "New Names"] for i in old_names_base_session_kept]
        new_names_session_1_kept = [name_change_record_session_1_df.loc[int(name_change_record_session_1_df.set_index('Old Names').index.get_loc(i)), "New Names"] for i in old_names_session_1_kept]

        print("CELLS REGISTERED")
        print(new_names_base_session_kept)
        print(new_names_session_1_kept)

        # need session names
        cellreg_registration_d = {
            f"{sessions[0]}" : new_names_base_session_kept,
            f"{sessions[1]}" : new_names_session_1_kept,
            
        }

        cellreg_df = pd.DataFrame.from_dict(cellreg_registration_d)

        cellreg_df.to_csv(f"{mouse_dir}/cellreg_{sessions[0]}_{sessions[1]}.csv", index=False)
    

if __name__ == "__main__":
    main()