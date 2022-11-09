import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from typing import List
from numpy.core.fromnumeric import mean
import pandas as pd
from scipy import stats
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

def find_paths(root_path: Path, middle: str, endswith: str) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, "**", middle, "**", endswith), recursive=True,
    )
    return files

def str_to_int(arr) -> list:
    new_arr = []
    for i in arr:
        if "-" in i:
            neg_num = i.replace("-","")
            neg_num = float(neg_num)
            neg_num = round(neg_num, 2)
            if neg_num == 0.00:
                # it's a zero
                new_arr.append(str(neg_num))
            else:
                neg_num = "-" + str(neg_num)
                new_arr.append(neg_num)
        else:
            i = float(i)
            i = round(i, 2)
            i = str(i)
            new_arr.append(i)

    return new_arr

def add_val(arr, val):
    
    return np.append(arr, [val])

def plot_trace_deco(plot_trace):

    def wrapper(x,y,every_nth,xlabel,ylabel,title):
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_xticks(x)
        ax.set_xlabel(xlabel)
        for n, label in enumerate(ax.xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)
        for n, label in enumerate(ax.xaxis.get_major_ticks()):
            if n % every_nth != 0:
                label.set_visible(False)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    return wrapper

@plot_trace_deco
def plot_trace(x, y, every_nth, xlabel, ylabel):
    plt.show()
    plt.close()

def zscore(obs_value, mu, sigma):
    return (obs_value - mu) / sigma

def main():

    session_root = r"/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis"

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

    for combo in list_of_combos_we_care_about:
        #combo = "Block_Trial_Type_Reward_Size_Start_Time_(s)"

        #for session in sessions:
        files = find_paths(session_root, f"{combo}","speeds_z_-5_5.csv")

        for csv in files:
            print("CURR:",csv)
            df = pd.read_csv(csv)
            fig, ax = plt.subplots()
            t = add_val(str_to_int(list(df.columns)[1:]), "5.0")
            every_nth = 30
            title = f"Z-scored, Savitzky (n={len(df)})"

            for row_idx in range(len(df)):
                arr_no_nan = list(df.iloc[row_idx,1:])

                xlabel = "Time from trigger (s)"
                ylabel = "Speed (cm/s)"

                # Savitzky - Z-score
                # make sure to not have any nans in the arr
                sav_z_arr = savgol_filter(arr_no_nan, window_length=33, polyorder=2)
                # change the df
                df.iloc[row_idx,1:] = sav_z_arr
                ax.plot(t, add_val(sav_z_arr, np.nan))

                """# Gaussian - Z-score
                gaus_z_arr = gaussian_filter1d(z_arr, sigma = 1.5)
                plot_trace(t, gaus_z_arr, 30, xlabel, ylabel, title="Z-scored, Gaussian")"""
            
            ax.set_xticks(t)
            ax.set_xlabel(xlabel)
            for n, label in enumerate(ax.xaxis.get_ticklabels()):
                if n % every_nth != 0:
                    label.set_visible(False)
            for n, label in enumerate(ax.xaxis.get_major_ticks()):
                if n % every_nth != 0:
                    label.set_visible(False)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            plt.savefig(csv.replace(".csv","savgol.png"))
            plt.close()

            df.to_csv(csv.replace(".csv","savgol.csv"), index=False)

if __name__ == "__main__":
    main()