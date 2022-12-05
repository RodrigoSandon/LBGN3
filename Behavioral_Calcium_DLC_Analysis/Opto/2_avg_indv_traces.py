import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from typing import List
from numpy.core.fromnumeric import mean
import pandas as pd
from pathlib import Path
from scipy import stats

def make_avg_speed_table(filename, csv_path, out_filename, half_of_time_window):
    step: float
    fps: int

    df_speed = pd.read_csv(csv_path)
    #print(df_speed.T.head())

    df_speed.columns = df_speed.iloc[0]
    df_speed = df_speed.iloc[1:, :]
    df_speed = df_speed.iloc[:, 1:]

    # Get the length of the time column currently
    time_col = df_speed.T.index.values.tolist()
    length_time_col = len(time_col)
    print("len of time col: ",length_time_col)


    # most likely it will be - 1 the supposed length
    if length_time_col == 299:
        fps = 30
    if length_time_col == 599:
        fps = 60
    if length_time_col == 1198:
        fps = 120

    step = 1/fps
    # 0.03333 is due to the 30Hz
    # So fps is variable now, but how can we know it's variable ahead of time?
    x_axis = np.arange(-half_of_time_window, half_of_time_window, step).tolist()
    #print(len(x_axis))
    #x_axis = [round(i, 1) for i in x_axis]
    if length_time_col == 1198:
        x_axis = x_axis[:-2]
    else:
        x_axis = x_axis[:-1]
    
    print("length of time axis: ",len(x_axis))
    
    avg_of_col_speed_lst = []
    for col_name, col_data in df_speed.iteritems():
        #print(list(df_speed[col_name].dropna()))
        #print([type(i) for i in list(df_speed[col_name].dropna())])
        timepoint_avg = stats.tmean(list(df_speed[col_name].dropna()))
        avg_of_col_speed_lst.append(timepoint_avg)

    #print("here:", len(x_axis), len(avg_of_col_speed_lst))
    #print(avg_of_col_speed_lst)
    #types = [type(i) for i in avg_of_col_speed_lst]
    #print(types)

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


    session_root = r"/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis_2/"

    """list_of_combos_we_care_about = [
            "Block_Start_Time_(s)",
            "Block_Omission_Start_Time_(s)",
            "Block_Reward_Size_Start_Time_(s)",
            "Block_Reward_Size_Shock_Ocurred_Start_Time_(s)",
            "Block_Shock_Ocurred_Start_Time_(s)",
            "Block_Trial_Type_Start_Time_(s)",
            "Shock_Ocurred_Start_Time_(s)",
            "Trial_Type_Start_Time_(s)",
            "Trial_Type_Reward_Size_Start_Time_(s)",
            "Block_Trial_Type_Omission_Start_Time_(s)",
            "Block_Trial_Type_Reward_Size_Start_Time_(s)",
            "Block_Trial_Type_Shock_Ocurred_Start_Time_(s)",
            "Block_Trial_Type_Win_or_Loss_Start_Time_(s)",
            "Trial_Type_Shock_Ocurred_Start_Time_(s)",
            "Win_or_Loss_Start_Time_(s)",
            "Block_Win_or_Loss_Start_Time_(s)",
            "Learning_Stratergy_Start_Time_(s)",
            "Omission_Start_Time_(s)",
            "Reward_Size_Start_Time_(s)",
        ]"""

    list_of_combos_we_care_about = [

            "Block_Trial_Type_Start_Time_(s)",
        ]

    filename = "speeds_z_-5_5_savgol.csv"

    for combo in list_of_combos_we_care_about:
        files = find_paths(session_root, combo ,filename)


        for csv in files:

            print(f"CURR CSV: {csv}")
            df: pd.DataFrame
            df = pd.read_csv(csv)
            trial_num = len(df)
        
            new_path = make_avg_speed_table(filename, csv_path=csv, out_filename="speeds_z_-5_5_savgol_avg.csv", half_of_time_window=5)
            plot_avg_speed(csv_path=new_path, event_num=trial_num)

def one_process():
    csv = "/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis_2/BLA_NAcShell/ArchT/Choice/RRD21/body/RRD21_choice_AlignmentData/Block_Trial_Type_Start_Time_(s)/(1.0, 'Free')/speeds_z_-5_5_savgol.csv"
    filename = "speeds_z_-5_5_savgol.csv"
    print(f"CURR CSV: {csv}")
    df: pd.DataFrame
    df = pd.read_csv(csv)
    trial_num = len(df)

    new_path = make_avg_speed_table(filename, csv_path=csv, out_filename="speeds_z_-5_5_savgol_avg.csv", half_of_time_window=5)
    plot_avg_speed(csv_path=new_path, event_num=trial_num)

if __name__ == "__main__":
    main()
    #one_process()
