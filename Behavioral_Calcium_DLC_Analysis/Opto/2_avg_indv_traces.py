import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from typing import List
from numpy.core.fromnumeric import mean
import pandas as pd
from pathlib import Path

def make_avg_speed_table(filename, csv_path, out_filename, half_of_time_window):

    df_speed = pd.read_csv(csv_path)

    df_speed.columns = df_speed.iloc[0]
    df_speed = df_speed.iloc[1:, :]
    df_speed = df_speed.iloc[:, 1:]

    # 0.03333 is due to the 30Hz
    x_axis = np.arange(-half_of_time_window, half_of_time_window, 0.03333).tolist()
    #x_axis = [round(i, 1) for i in x_axis]
    x_axis = x_axis[:-1]
    
    avg_of_col_speed_lst = []
    for col_name, col_data in df_speed.iteritems():
        timepoint_avg = df_speed[col_name].mean()
        avg_of_col_speed_lst.append(timepoint_avg)

    #print("here:", len(x_axis), len(avg_of_col_speed_lst))

    csv_prep_unnorm = {
        "Time_(s)" : x_axis,
        "Avg_Speed_(cm/s)" : avg_of_col_speed_lst
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
    ax.set_title(f"Avg. Speed of Trials, Z-scored, Savitzky (n={event_num})")
    fig.savefig(csv_path.replace(".csv",".png"))
    plt.close(fig)

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
    
    # OR

    session_root = r"/media/rory/RDT VIDS/BORIS/"

    combo = "Block_Trial_Type_Reward_Size_Start_Time_(s)"

    for session in sessions:
        filename = "speeds_z_-5_5savgol.csv"
        files = find_paths(session, f"{combo}",filename)

        for csv in files:

            print(f"CURR CSV: {csv}")
            df: pd.DataFrame
            df = pd.read_csv(csv)
            trial_num = len(df)
        
            new_path = make_avg_speed_table(filename, csv_path=csv, out_filename="speeds_z_-5_5savgol_avg.csv", half_of_time_window=5)
            plot_avg_speed(csv_path=new_path, event_num=trial_num)

if __name__ == "__main__":
    main()