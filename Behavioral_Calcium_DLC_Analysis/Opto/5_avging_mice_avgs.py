import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from typing import List
from numpy.core.fromnumeric import mean
import pandas as pd
from pathlib import Path


def mean_of_lst_o_lsts(df: pd.DataFrame, col_no_include) -> list:
    # this df does contain the event # column
    num_of_points_per_timepoint = len(list(df.columns)) - 1

    d = {}
    for col in df:
        d[col] = df[col]

    lists_to_avg = []
    for key in d:
        if key != col_no_include:
            #print(d[key])
            lists_to_avg.append(d[key])
    
    zipped = zip(*lists_to_avg)
    #print(zipped)

    avg_lst = []

    for index, tuple in enumerate(zipped):
        #print(tuple)
        avg_o_timepoint = sum(list(tuple)) / num_of_points_per_timepoint
        avg_lst.append(avg_o_timepoint)
    
    return avg_lst

def make_avg_speed_table(filename, csv_path, out_filename):

    df_speed = pd.read_csv(csv_path)
    
    avg_list = mean_of_lst_o_lsts(df_speed, "Time_(s)")

    csv_prep_unnorm = {
        "Time_(s)" : list(df_speed["Time_(s)"]),
        "Avg_Speed_(cm/s)" : avg_list
    }

    path_to_save = csv_path.replace(filename, out_filename)
    dff_n_speed = pd.DataFrame.from_dict(csv_prep_unnorm)
    dff_n_speed.to_csv(path_to_save, index = False)

    return path_to_save

def add_val(arr, val):
    
    return np.append(arr, [val])

def plot_avg_speed(csv_path, event_num):
    """Plots the figure from the csv file given"""
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots()
    every_nth = 30
    # add the last value

    t = list(df["Time_(s)"]) + [5.0]
    y = add_val(np.array(list(df["Avg_Speed_(cm/s)"])), np.nan)

    ax.plot(t, y)
    ax.set_xticks([round(i, 1) for i in t])
    ax.set_xlabel("Time from trigger (s)")
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    for n, label in enumerate(ax.xaxis.get_major_ticks()):
        if n % every_nth != 0:
            label.set_visible(False)
    ax.set_ylabel("Average speed (cm/s)")
    ax.set_title(f"Avg. Speed of Mice, Z-scored, Savitzky (n={event_num})")
    fig.savefig(csv_path.replace(".csv",".png"))
    plt.close(fig)

def find_paths(root_path: Path, mid, endswith: str) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, "**", mid, "**", endswith), recursive=True,
    )
    return files

def main():


    session_root = r"/media/rory/RDT VIDS/BetweenMiceAlignmentData"

    combo = "Block_Trial_Type_Reward_Size_Start_Time_(s)"


    filename = "all_speeds_z_-5_5savgol_avg.csv"
    files = find_paths(session_root, f"{combo}",filename)

    for csv in files:

        print(f"CURR CSV: {csv}")
        df: pd.DataFrame
        df = pd.read_csv(csv)
        trial_num = len(list(df.columns)) - 1

        new_path = make_avg_speed_table(filename, csv_path=csv, out_filename="avg_all_speeds_z_-5_5savgol_avg.csv")
        plot_avg_speed(csv_path=new_path, event_num=trial_num)

if __name__ == "__main__":
    main()