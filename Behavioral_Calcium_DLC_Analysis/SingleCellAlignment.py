import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from typing import List, Optional
from numpy.core.fromnumeric import mean
import pandas as pd
import seaborn as sns
from scipy import stats
import Cell
from operator import attrgetter
from pathlib import Path
from scipy.ndimage import gaussian_filter1d


def avg_cell_eventrace(df, csv_path, cell_name, plot: bool, export_avg: bool, filename: str, ending: str):
    """Plots the figure from the csv file given"""
    path_to_save = csv_path.replace(
        f"{filename}.csv", f"avg_{filename}{ending}.png")
    #df_sub = df.iloc[:, 1:]
    # print(df_sub.head())
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
            f"{filename}.csv", f"avg_{filename}{ending}.csv")
        export_avg_cell_eventraces(cell_name, avg_of_col_lst, path_to_save)


def export_avg_cell_eventraces(
    cell_name, avg_dff_list_for_timewindow_n_event, out_path
):
    df = pd.DataFrame(avg_dff_list_for_timewindow_n_event, columns=[cell_name])
    df.to_csv(out_path, index=False)


def zscore(obs_value, mu, sigma):
    return (obs_value - mu) / sigma

# limit idx, should be total range that your zscoring, doesn't indicate baseline
def custom_standardize_limit_fixed(
        df: pd.DataFrame, baseline_min, baseline_max, limit_idx):
    """A limit indicates when to stop z-scoring based off of the baseline."""
    for col in df.columns:
        subwindow = list(df[col])[baseline_min: baseline_max + 1]

        mean_for_cell = stats.tmean(subwindow)
        stdev_for_cell = stats.tstd(subwindow)

        new_col_vals = []
    
        for count, ele in enumerate(list(df[col])):
            if count >= baseline_min and count <= limit_idx:
                z_value = zscore(ele, mean_for_cell, stdev_for_cell)
            else:  # if outside limits of zscoring, don't zscore
                z_value = ele
            new_col_vals.append(z_value)

        df[col] = new_col_vals
    return df
# case 1: /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/SingleCellAlignmentData/C01/Shock Ocurred_Choice Time (s)/True/plot_ready.csv
# case 2: /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/RDT D1/SingleCellAlignmentData/C01/Shock Ocurred_Choice Time (s)/True/plot_ready.csv
# - nvm, the same, but then transfer all cells into corresponding session/event into betweencellalignment
# betweencellalignment already covered, just need to output name for this step


def find_paths(root_path: Path, middle: str, endswith: str) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, "**", middle, "**", endswith), recursive=True,
    )
    return files

def find_paths_2mids(root_path: Path, middle_1: str, middle_2: str, endswith: str) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, "**", middle_1,"**", middle_2, "**", endswith), recursive=True,
    )
    return files

def gaussian_smooth(df, sigma: float = 1.5):
    # df = df.iloc[:, 1:]  # omit first col

    return df.apply(gaussian_filter1d, sigma=sigma, axis=0)

def main():
    MASTER_ROOT = r"/media/rory/Padlock_DT/BLA_Analysis"
    mice = [
        "BLA-Insc-14",
        "BLA-Insc-15",
        "BLA-Insc-16",
        ]
    sessions = ["Pre-RDT RM"]
    event = "Block_Reward Size_Shock Ocurred_Choice Time (s)"
    ending_desired = "_z_fullwindow"
    filename = "plot_ready"

    # further preprocess the plot_ready files
    for mouse in mice:
        for session in sessions:
            files = find_paths_2mids(MASTER_ROOT, f"{mouse}/{session}/SingleCellAlignmentData", event ,f"{filename}.csv")

            for csv in files:
                #picking out certain events
                #if "Start Time (s)_Collection Time (s)" in csv:
          
                try:
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

                    # 1) Zscore
                    df = custom_standardize_limit_fixed(
                        df,
                        baseline_min=0,
                        baseline_max=200,
                        limit_idx=200
                    )
                    df = df.T

                    df = gaussian_smooth(df)

                    # 2) Average Z score per each trial
                    avg_cell_eventrace(
                        df, csv, cell_name, plot=True, export_avg=True, filename=filename, ending=ending_desired
                    )

                    df.insert(0, "Event #", col_to_save)

                    csv_moded_out_path = csv.replace(".csv", f"{ending_desired}.csv")
                    df.to_csv(csv_moded_out_path, index=False)
                except TypeError as e:
                    print(e)
                    pass

def onecell():

    files = [

        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-14/Pre-RDT RM/SingleCellAlignmentData/C04/Block_Reward Size_Choice Time (s)/(2.0, 'Large')/plot_ready.csv",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-15/Pre-RDT RM/SingleCellAlignmentData/C02/Block_Reward Size_Choice Time (s)/(2.0, 'Large')/plot_ready.csv",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-16/Pre-RDT RM/SingleCellAlignmentData/C03/Block_Reward Size_Choice Time (s)/(2.0, 'Large')/plot_ready.csv"


    ]
    ending_desired = "_z_fullwindow"

    for csv in files:
        #picking out certain events
        #if "Start Time (s)_Collection Time (s)" in csv:
    
        try:
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

            # 1) Zscore
            df = custom_standardize_limit_fixed(
                df,
                baseline_min=0,
                baseline_max=200,
                limit_idx=200
            )
            df = df.T

            def gaussian_smooth(df, sigma: float = 1.5):
                from scipy.ndimage import gaussian_filter1d
                # df = df.iloc[:, 1:]  # omit first col

                return df.apply(gaussian_filter1d, sigma=sigma, axis=0)
            df = gaussian_smooth(df)

            # 2) Average Z score per each trial
            avg_cell_eventrace(
                df, csv, cell_name, plot=True, export_avg=True, ending=ending_desired
            )

            df.insert(0, "Event #", col_to_save)

            csv_moded_out_path = csv.replace(".csv", f"{ending_desired}.csv")
            df.to_csv(csv_moded_out_path, index=False)
        except TypeError as e:
            print(e)
            pass

if __name__ == "__main__":
    main()
    #onecell()