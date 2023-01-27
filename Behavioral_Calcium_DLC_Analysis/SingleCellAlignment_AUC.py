import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from typing import List, Optional
from numpy.core.fromnumeric import mean
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import auc
import Cell
from operator import attrgetter
from pathlib import Path
from sklearn import metrics


def subwindow_auc(df: pd.DataFrame, x_coords: list, min, max):
    auc_list = []
    for col in df:
        subwindow = list(df[col])[min:max + 1]
        auc = metrics.auc(x_coords[min:max + 1], subwindow)
        auc_list.append(auc)

    # len of auc_list should correspond to number of trials
    return auc_list

# case 1: /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/SingleCellAlignmentData/C01/Shock Ocurred_Choice Time (s)/True/plot_ready.csv
# case 2: /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/RDT D1/SingleCellAlignmentData/C01/Shock Ocurred_Choice Time (s)/True/plot_ready.csv
# - nvm, the same, but then transfer all cells into corresponding session/event into betweencellalignment
# betweencellalignment already covered, just need to output name for this step


def find_paths(root_path: Path, middle: str, endswith: str) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, "**", middle, "**", endswith), recursive=True,
    )
    return files

def wilcoxon_analysis(postchoice_list: list, prechoice_list: list, total_num_cells: int, alpha) -> str:

        result_greater = stats.mannwhitneyu(
            postchoice_list, prechoice_list, alternative="greater"
        )

        result_less = stats.mannwhitneyu(
            postchoice_list, prechoice_list, alternative="less"
        )

        id = None
        if result_greater.pvalue < (alpha/total_num_cells):
            id = "+"
        elif result_less.pvalue < (alpha/total_num_cells):
            id = "-"
        else:
            id = "Neutral"

        return id

def main():
    # ex: /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/SingleCellAlignmentData/C18/Shock Ocurred_Choice Time (s)/True/plot_ready_z_fullwindow.csv
    # ex: /media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/RDT D1/Shock Ocurred_Choice Time (s)/True/all_concat_cells_z_fullwindow_id_auc.csv
    MASTER_ROOT = r"/media/rory/Padlock_DT/BLA_Analysis"
    """mice = [
        "BLA-Insc-1",
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
        "BLA-Insc-19",
        ]"""

    mice = [
        "BLA-Insc-14",
        "BLA-Insc-15",
        "BLA-Insc-16",
        ]

    sessions = ["Pre-RDT RM"]

    for mouse in mice:
        print(mouse)
        for session in sessions:
            files = find_paths(MASTER_ROOT, f"{mouse}/{session}/SingleCellAlignmentData","plot_ready_z_fullwindow.csv")
            print(session)
            for csv in files:
                try: 
                    #print(f"CURR CSV: {csv}")
                    ######## Get total number of cells that are going to be concatenated at these similar conditions: session, event, subevent ########
                    event = csv.split("/")[10]
                    subevent = csv.split("/")[11]
                    #### ONLY DOING IT FOR A SPECIFIC EVENT ###
                    if event == "Block_Reward Size_Shock Ocurred_Choice Time (s)":
                        #### ONLY FOR GETTING TOTAL NUMBER OF CELLS PURPOSES BEFOREHAND ####
                        corresponding_all_concat_cells_csv = f"{MASTER_ROOT}/BetweenMiceAlignmentData/{session}/{event}/{subevent}/all_concat_cells_z_fullwindow.csv"
                        corresponding_all_concat_cells_df = pd.read_csv(corresponding_all_concat_cells_csv)
                        total_num_cells = len(list(corresponding_all_concat_cells_df.columns))
                        #print(total_num_cells)
                        #### ONLY FOR GETTING TOTAL NUMBER OF CELLS PURPOSES BEFOREHAND ####

                        cell_name = csv.split("/")[9]
                        df: pd.DataFrame
                        df = pd.read_csv(csv)
                        # print(df.head())
                        # save col that u will omit once transposed
                        col_to_save = list(df["Event #"])
                        df = df.T
                        df = df.iloc[1:, :]  # omit first row

                        # print(df.head())
                        # one col is 1 trial now
                        x_coords = list(range(len(df)))
                        
                        prechoice_min = 71
                        prechoice_max = 101
                        postchoice_min = 101
                        postchoice_max = 131
                        auc_prechoice = subwindow_auc(df, x_coords, prechoice_min, prechoice_max) 
                        auc_postchoice = subwindow_auc(df, x_coords, postchoice_min, postchoice_max) 

                        d_to_save = {
                            "Trial #": col_to_save,
                            "Prechoice AUC": auc_prechoice,
                            "Postchoice AUC": auc_postchoice
                        }

                        auc_df = pd.DataFrame.from_records(d_to_save, index=range(len(col_to_save)))
                        auc_df_out = csv.replace(".csv", f"_auc_info_{prechoice_min}_{prechoice_max}_{postchoice_min}_{postchoice_max}.csv")
                        auc_df.to_csv(auc_df_out, index = False)
                        
                        alpha = 0.05
                        id = wilcoxon_analysis(auc_postchoice, auc_prechoice, total_num_cells, alpha)

                        id_d = {cell_name : id}
                        id_df = pd.DataFrame.from_records(id_d, index=[0])
                        id_df_out = csv.replace("plot_ready_z_fullwindow.csv", f"id_z_fullwindow_auc_bonf{alpha}_{prechoice_min}_{prechoice_max}_{postchoice_min}_{postchoice_max}.csv")
                        id_df.to_csv(id_df_out, index=False)

                except FileNotFoundError as e:
                    print(e)
                    pass

if __name__ == "__main__":
    main()