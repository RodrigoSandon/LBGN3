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
    sessions = ["RDT D1"]

    for mouse in mice:
        print(mouse)
        for session in sessions:
            files = find_paths(MASTER_ROOT, f"{mouse}/{session}/SingleCellAlignmentData","avg_plot_ready_z_fullwindow.csv")
            print(session)
            for csv in files:
                #print(f"CURR CSV: {csv}")
                ######## Get total number of cells that are going to be concatenated at these similar conditions: session, event, subevent ########
                event = csv.split("/")[10]
                subevent = csv.split("/")[11]

                cell_name = csv.split("/")[9]
                df: pd.DataFrame
                df = pd.read_csv(csv)

                x_coords = list(range(len(df)))
                
                prechoice = list(df[cell_name])[0:101 + 1]
                postchoice = list(df[cell_name])[101:121 + 1]
                
                alpha = 0.01
                id = wilcoxon_analysis(postchoice, prechoice, alpha)

                id_d = {cell_name : id}
                id_df = pd.DataFrame.from_records(id_d, index=[0])
                id_df_out = csv.replace("avg_plot_ready_z_fullwindow.csv", f"id_z_fullwindow_alpha{alpha}_-10_0_0_2.csv")
                id_df.to_csv(id_df_out, index=False)

if __name__ == "__main__":
    main()