import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from typing import List, Optional
from numpy.core.fromnumeric import mean
import pandas as pd
import seaborn as sns
from scipy import stats
from operator import attrgetter
from pathlib import Path


def avg_cell_eventrace(df, csv_path, plot: bool, export_avg: bool):
    """Plots the figure from the csv file given"""
    path_to_save = csv_path.replace(
        "speeds.csv", "speeds_z_-5_5.png")
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
        plt.title(("Average Speed for Event Window"))
        plt.xlabel("Time from trigger (s)")
        plt.ylabel("Average Speed (n=%s)" % (row_count))
        plt.savefig(path_to_save)
        plt.close()

    if export_avg == True:
        path_to_save = csv_path.replace(
            "speeds.csv", "avg_speed_z_-5_5.csv")
        export_avg_cell_eventraces(avg_of_col_lst, path_to_save)


def export_avg_cell_eventraces(
 avg_dff_list_for_timewindow_n_event, out_path
):
    df = pd.DataFrame(avg_dff_list_for_timewindow_n_event, columns=["Speed_(cm/s)"])
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


def main():

    sessions = [
        "/media/rory/RDT VIDS/BORIS/RRD170/RDT OPTO CHOICE 0115",
        "/media/rory/RDT VIDS/BORIS/RRD168/RDT OPTO CHOICE 0114",
        "/media/rory/RDT VIDS/BORIS/RRD171/RDT OPTO CHOICE 0104",
        "/media/rory/RDT VIDS/BORIS/RRD81/RDT OPTO CHOICE 1104"
        ]

    combo = "Block_Trial_Type_Reward_Size_Start_Time_(s)"

    for session in sessions:
        files = find_paths(session, f"{combo}","speeds.csv")

        for csv in files:

            try:
                print(f"CURR CSV: {csv}")
                df: pd.DataFrame
                df = pd.read_csv(csv)
            
                col_to_save = list(df["Event_#"])
                df = df.T
                df = df.iloc[1:, :]  # omit first row

                # print(df.head())

                # 1) Zscore
                df = custom_standardize_limit_fixed(
                    df,
                    baseline_min=0,
                    baseline_max=300,
                    limit_idx=300
                )
                df = df.T

                def gaussian_smooth(df, sigma: float = 1.5):
                    from scipy.ndimage import gaussian_filter1d
                    # df = df.iloc[:, 1:]  # omit first col

                    return df.apply(gaussian_filter1d, sigma=sigma, axis=0)
                df = gaussian_smooth(df)

                # 2) Average Z score per each trial
                avg_cell_eventrace(
                    df, csv, plot=True, export_avg=True
                )

                df.insert(0, "Event_#", col_to_save)

                csv_moded_out_path = csv.replace(".csv", "_z_-5_5.csv")
                df.to_csv(csv_moded_out_path, index=False)
            except TypeError as e:
                print(e)
                pass

if __name__ == "__main__":
    main()