import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
from typing import List
import json

def find_avg_dff_of_cell_for_event(session_path, startswith):

    files = glob.glob(
        os.path.join(session_path, "**", "%s*") % (startswith),
        recursive=True,
    )

    return files


def create_concat_csv(lst_of_all_avg_cell_csv_paths, root_path):
    event_dict = {}
    
    # OLD example: /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-5/Session-20210510-093930_BLA-Insc-5_RM_D1/SingleCellAlignmentData/C01/Block_Choice Time (s)
    # NEW example 11/24/21: /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/Late Shock D1/SingleCellAlignmentData/C01/Block_Learning Stratergy_Choice Time (s)/(3.0, 'Win Stay')/avg_plot_ready.csv
    # REPLACED BY: /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/Late Shock D1/BetweenCellAlignmentData
    # for the avg cell csv paths,
    for avg_cell_csv_path in lst_of_all_avg_cell_csv_paths:
        cell_name = avg_cell_csv_path.split("/")[9]
        combo = avg_cell_csv_path.split("/")[10]
        subcombo = avg_cell_csv_path.split("/")[11]
        # if combo doesn't exist, but you still have to account for this current avg ready csv
        if combo not in event_dict:
            event_dict[combo] = {}
            event_dict[combo][subcombo] = {}
            avg_dff_traces_df = pd.read_csv(avg_cell_csv_path)
            event_dict[combo][subcombo][cell_name] = avg_dff_traces_df[
                cell_name
            ].tolist()
        # if combo exists
        elif combo in event_dict:
            # if subcombo doesn't exist
            if subcombo not in event_dict[combo]:
                event_dict[combo][subcombo] = {}
                avg_dff_traces_df = pd.read_csv(avg_cell_csv_path)
                event_dict[combo][subcombo][cell_name] = avg_dff_traces_df[
                    cell_name
                ].tolist()
                # if subcombo does exist
            elif subcombo in event_dict[combo]:
                # so a list exists already, there isn't repeats of cells
                # (which is the only thing these two csv paths should be differing in),
                # so you can ignore the cell check
                avg_dff_traces_df = pd.read_csv(avg_cell_csv_path)
                event_dict[combo][subcombo][cell_name] = avg_dff_traces_df[
                    cell_name
                ].tolist()

    # now have every dff trace in their proper category
    for combo in event_dict:
        for subcombo in event_dict[combo]:
            # now have all cells for combo, now just combine them all to one csv
            # print(event_dict[event][combo])
            concatenated_cells_df = pd.DataFrame.from_dict(
                event_dict[combo][subcombo])
            new_path = os.path.join(root_path, combo, subcombo)
            os.makedirs(new_path, exist_ok=True)
            concatenated_cells_df.to_csv(
                os.path.join(new_path, "concat_cells_id_z_fullwindow_auc_bonf0.05_71_101_101_131.csv"), index=False
            )


def main():
    lst = [
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5"
    ]
    for i in lst:

        MOUSE_BATCH_PATH = Path(i)

        """session_types = [
            "PR D1",
            "PR D2",
            "Pre-RDT RM",
            "RDT D1",
            "RDT D2",
            "RDT D3",
            "Post-RDT D1",
            "Post-RDT D2",
            "Post-RDT D3",
            "RM D1",
            "RM D2",
            "RM D3",
            "RM D8",
            "RM D9",
            "RM D10",
            "Late Shock D1",
            "Late Shock D2",
        ]  # UPDATE 1/3/22 -> SHOCK SESSIONS ARE PROCESSED IN ANOTHER PY FILE"""
        session_types = [
        
            "RDT D1",
            
        ]  # UPDATE 1/3/22 -> SHOCK SESSIONS ARE PROCESSED IN ANOTHER PY FILE


        for root, dirs, files in os.walk(MOUSE_BATCH_PATH):
            for dir_name in dirs:

                # caveats
                if dir_name == "RDT D2 NEW_SCOPE":
                    dir_name = "RDT D2"
                elif dir_name == "RDT D3 NEW_SCOPE":
                    dir_name = "RDT D3"
                elif dir_name == "RM D8 TANGLED":
                    dir_name = "RM D8"
                elif dir_name == "Shock Test NEW_SCOPE":
                    dir_name = "Shock Test"

                for ses_type in session_types:
                    if (
                        dir_name == ses_type
                    ):  # means ses type string was found in dirname
                        print(f"Session type: {ses_type}, Found: {dir_name}")
                        SESSION_PATH = os.path.join(root, dir_name)
                        print(f"Working on... {SESSION_PATH}")
                        lst_of_avg_cell_csv_paths_for_session = (
                            find_avg_dff_of_cell_for_event(
                                SESSION_PATH, "id_z_fullwindow_auc_bonf0.05_71_101_101_131.csv"
                            )
                        )
                        # file.write("\n".join(lst_of_avg_cell_csv_paths_for_session))
                        bw_cell_alignment_folder_name = "BetweenCellAlignmentData"
                        bw_cell_data_path = os.path.join(
                            SESSION_PATH, bw_cell_alignment_folder_name
                        )
                        # os.makedirs(bw_cell_data_path, exist_ok=True)
                        # now in this bw_cell_data_path, make everything, not anywhere else
                        create_concat_csv(
                            lst_of_avg_cell_csv_paths_for_session, bw_cell_data_path
                        )


if __name__ == "__main__":
    main()