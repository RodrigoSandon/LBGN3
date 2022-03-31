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

def avg_cell_eventrace(df, csv_path, cell_name, plot: bool, export_avg: bool):
    """Plots the figure from the csv file given"""
    path_to_save = csv_path.replace(
        "plot_ready.csv", "avg_plot_z_fullwindow.png")
    xaxis = list(df.columns)

    row_count = len(df)

    avg_of_col_lst = []
    for col_name, col_data in df.iteritems():
        if stats.tmean(list(df[col_name])) > 10000:
            print(col_name)
            print(list(df[col_name]))
        avg_dff_of_timewindow_of_event = stats.tmean(list(df[col_name]))
        avg_of_col_lst.append(avg_dff_of_timewindow_of_event)

    if plot == True:

        plt.plot(xaxis, avg_of_col_lst)
        plt.title(("Average DF/F Trace for %s Event Window") % (cell_name))
        plt.xlabel("Time (s)")
        plt.ylabel("Average DF/F (n=%s)" % (row_count))
        plt.savefig(path_to_save)
        plt.close()

    if export_avg == True:
        path_to_save = csv_path.replace(
            "plot_ready.csv", "avg_plot_ready_z_fullwindow.csv")
        export_avg_cell_eventraces(cell_name, avg_of_col_lst, path_to_save)


def export_avg_cell_eventraces(
    cell_name, avg_dff_list_for_timewindow_n_event, out_path
):
    df = pd.DataFrame(avg_dff_list_for_timewindow_n_event, columns=[cell_name])
    df.to_csv(out_path, index=False)


def subwindow_auc(df: pd.DataFrame, x_coords: list, min, max, auc_list: list):

    for col in df:
        subwindow = list(df[col])[min:max + 1]
        auc = metrics.auc(x_coords, subwindow)
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

def wilcoxon_analysis(postchoice_list: list, prechoice_list: list, alpha = 0.01) -> str:

        result_greater = stats.mannwhitneyu(
            postchoice_list, prechoice_list, alternative="greater"
        )

        result_less = stats.mannwhitneyu(
            postchoice_list, prechoice_list, alternative="less"
        )

        id = None
        if result_greater.pvalue < (alpha / len(prechoice_list)):
            id = "+"
        elif result_less.pvalue < (alpha / len(prechoice_list)):
            id = "-"
        else:
            id = "Neutral"

        return id

def main():
    MASTER_ROOT = r"/media/rory/Padlock_DT/BLA_Analysis"
    mice = ["BLA-Insc-14","BLA-Insc-15","BLA-Insc-16","BLA-Insc-18","BLA-Insc-19",]
    sessions = ["RDT D1", "RDT D2", "RDT D3"]

    for mouse in mice:
        for session in sessions:
            files = find_paths(MASTER_ROOT, f"{mouse}/{session}/SingleCellAlignmentData","plot_ready_z_fullwindow.csv")

            for csv in files:
                print(f"CURR CSV: {csv}")
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
                
                auc_prechoice = subwindow_auc(df, x_coords, 21, 51) # -8 to -5
                auc_postchoice = subwindow_auc(df, x_coords, 101, 131) # 0 to 3

                d_to_save = {
                    "Trial #": col_to_save,
                    "Prechoice AUC": auc_prechoice,
                    "Postchoice AUC": auc_postchoice
                }

                auc_df = pd.DataFrame.from_records(d_to_save)
                auc_df_out = csv.replace(".csv", "_aucs.csv")
                auc_df.to_csv(auc_df_out, index = False)
                
                id = wilcoxon_analysis(auc_postchoice, auc_prechoice)

                id_d = {cell_name : id}
                id_df = pd.DataFrame.from_records(id_d)
                id_df_out = csv.replace("plot_ready_z_fullwindow.csv", "id_fullwindow_z_auc.csv")
                id_df.to_csv(id_df_out, index=False)

if __name__ == "__main__":
    main()