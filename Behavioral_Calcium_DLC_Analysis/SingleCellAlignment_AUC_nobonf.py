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

def wilcoxon_analysis(postchoice_list: list, prechoice_list: list, alpha) -> str:

        result_greater = stats.mannwhitneyu(
            postchoice_list, prechoice_list, alternative="greater"
        )

        result_less = stats.mannwhitneyu(
            postchoice_list, prechoice_list, alternative="less"
        )

        id = None
        if result_greater.pvalue < alpha:
            id = "+"
        elif result_less.pvalue < alpha:
            id = "-"
        else:
            id = "Neutral"

        return id

def main():
    # ex: /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/SingleCellAlignmentData/C18/Shock Ocurred_Choice Time (s)/True/plot_ready_z_fullwindow.csv
    # ex: /media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/RDT D1/Shock Ocurred_Choice Time (s)/True/all_concat_cells_z_fullwindow_id_auc.csv
    MASTER_ROOT = r"/media/rory/Padlock_DT/BLA_Analysis"
    mice = [
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
        ]
    sessions = ["Pre-RDT RM"]

    for mouse in mice:
        print(mouse)
        for session in sessions:
            files = find_paths(MASTER_ROOT, f"{mouse}/{session}/SingleCellAlignmentData","plot_ready_z_fullwindow.csv")
            print(session)
            for csv in files:
                #print(f"CURR CSV: {csv}")
                ######## Get total number of cells that are going to be concatenated at these similar conditions: session, event, subevent ########
                event = csv.split("/")[10]
                subevent = csv.split("/")[11]

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
                
                prechoice_min = 0
                prechoice_max = 51
                postchoice_min = 111
                postchoice_max = 141
                auc_prechoice = subwindow_auc(df, x_coords, 71, 101) # -8 to -5 | TO -3 TO 0 | -10 -5
                auc_postchoice = subwindow_auc(df, x_coords, 101, 131,) # 0 to 3 | changed to -3 to 0 5/5/22 | TO 0 TO 3

                d_to_save = {
                    "Trial #": col_to_save,
                    "Prechoice AUC": auc_prechoice,
                    "Postchoice AUC": auc_postchoice
                }

                auc_df = pd.DataFrame.from_records(d_to_save, index=range(len(col_to_save)))
                auc_df_out = csv.replace(".csv", "_auc_info.csv")
                auc_df.to_csv(auc_df_out, index = False)
                
                alpha = 0.05
                id = wilcoxon_analysis(auc_postchoice, auc_prechoice, alpha)

                id_d = {cell_name : id}
                id_df = pd.DataFrame.from_records(id_d, index=[0])
                id_df_out = csv.replace("plot_ready_z_fullwindow.csv", f"id_z_fullwindow_auc_{prechoice_min}_{prechoice_max}_{postchoice_min}_{postchoice_max}.csv")
                id_df.to_csv(id_df_out, index=False)

if __name__ == "__main__":
    main()