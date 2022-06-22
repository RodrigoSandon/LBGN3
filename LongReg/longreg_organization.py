import pandas as pd
import numpy as np
import glob, os
import shutil
import matplotlib.pyplot as plt

""" 
Goal:
    - To track cells across sessions
    - The priority will be on Pre-RDT RM, RDT D1, D2, D3
    - Make a graph that displays thefrequency of the number of aligned sessions out of all of these
    - And a separate graph including all session types, not just the ones listed above
"""

def find_paths_endswith(root_path, endswith) -> list:

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )

    return files

def new_to_old_name(cell_name: str, df: pd.DataFrame) -> str:

    new_name_row_idx = list(df[df["New Names"] == cell_name].index.values)[0]
    old_cell_name = df.iloc[new_name_row_idx, df.columns.get_loc("Old Names")]

    return old_cell_name

def old_to_new_name(cell_name: str, df: pd.DataFrame) -> str:

    old_name_row_idx = list(df[df["Old Names"] == cell_name].index.values)[0]
    new_cell_name = df.iloc[old_name_row_idx, df.columns.get_loc("New Names")]

    return new_cell_name

def main():


    ROOT = r"/media/rory/Padlock_DT/BLA_Analysis/"
    batch_names = ["PTP_Inscopix_#1", "PTP_Inscopix_#3", "PTP_Inscopix_#4", "PTP_Inscopix_#5"]
    events = ["Block_Reward Size_Shock Ocurred_Start Time (s)_Collection Time (s)"]
    
    """
    d = {mouse: {global_cell: }}

    Afterwards, you need to copy whatever is in those sessions paths to new session path (specify)
    """
    d = {}

    for folder_name in batch_names:
        BATCH_ROOT = os.path.join(ROOT, folder_name)
        mouse_paths = [
            os.path.join(BATCH_ROOT, dir)
            for dir in os.listdir(BATCH_ROOT)
            if os.path.isdir(os.path.join(BATCH_ROOT, dir))
            and dir.startswith("BLA")
        ]
        for mouse_path in mouse_paths:
            mouse = mouse_path.split("/")[6]
            print("CURRENT MOUSE: ", mouse)

            d[mouse] = {}

            try:
                mouse_longreg_csv_file = f"{mouse_path}/longreg_results_preprocessed.csv"
                print(mouse_longreg_csv_file)
                df_longreg = pd.read_csv(mouse_longreg_csv_file)
                
                #going through each row in longreg file
                for index, row in df_longreg.iterrows():
                    # pick up only certain sessions you wanna track

                    global_cell = df_longreg.iloc[index, df_longreg.columns.get_loc("global_cell_index")]
                    local_cell = df_longreg.iloc[index, df_longreg.columns.get_loc("local_cell_name")]
                    local_session = df_longreg.iloc[index, df_longreg.columns.get_loc("session_name")]

                    ncc_score = df_longreg.iloc[index, df_longreg.columns.get_loc("ncc_score")]
                    centroid_distance = df_longreg.iloc[index, df_longreg.columns.get_loc("centroid_distance")]

                    if local_session == "Pre-RDT RM" or local_session == "RDT D1" or local_session == "RDT D2" or local_session == "RDT D2":

                        if global_cell in d[mouse]:
                            d[mouse][global_cell]
                        else:
                            d[mouse][global_cell] = { local_cell: {
                                "session_name":local_session,
                                "ncc_score": ncc_score,
                                "centroid_distance": centroid_distance
                                                    }}
                        
                        # if I know local session, then I can do:
                        # /media/rory/Padlock_DT/BLA_Analysis/{batch}/{mouse}/{local_session}/SingleCellAlignmentData/{new_local_cell_name}/{event}
                        # "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/naming_change_record.csv"
                        
                        #every row is a new session, new local cell but same global cell (potentially)
                        records_csv = f"{BATCH_ROOT}/{mouse}/{local_session}/naming_change_record.csv"

                        records_df = pd.read_csv(records_csv)
                        
                        # the new local cell name starts at 1 and goes in subsequent order
                        new_local_cell_name = old_to_new_name(local_cell, records_csv)

                        #pull event you want to track
                        if len(events) == 1:
                            event = events[0]
                            src = f"{BATCH_ROOT}/{mouse}/{local_session}/SingleCellAlignmentData/{new_local_cell_name}/{event}"
                            os.makedirs(f"/media/rory/Padlock_DT/BLA_Analysis/LongReg/Results/{mouse}/{global_cell}/{local_session}/SingleCellAlignmentData/{new_local_cell_name}/", exist_ok=True)
                            dst = f"/media/rory/Padlock_DT/BLA_Analysis/LongReg/Results/{mouse}/{global_cell}/{local_session}/SingleCellAlignmentData/{new_local_cell_name}/{event}"

                            shutil.copytree(src, dst)


                        else:
                            for event in events:
                                pass




                        


            except FileNotFoundError as e:
                print(f"{mouse} does not have long reg results!")
                pass

    fig, ax = plt.subplots()

if __name__ == "__main__":
    main()