import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from typing import List
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from numpy.core.fromnumeric import mean
import seaborn as sns
from operator import attrgetter
from pathlib import Path

def find_paths(root_path: Path, middle: str, endswith: str) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, "**", middle, "**", endswith), recursive=True,
    )
    return files

def zscore(obs_value, mu, sigma):
    return (obs_value - mu) / sigma

# limit idx, should be total range that your zscoring, doesn't indicate baseline
def custom_standardize_limit_fixed(
        df: pd.DataFrame, baseline_min, baseline_max, limit_idx):
    """A limit indicates when to stop z-scoring based off of the baseline."""
    for col in df.columns:
        subwindow = list(df[col])[baseline_min: baseline_max + 1]

        col_mean = stats.tmean(subwindow)
        col_stdev = stats.tstd(subwindow)

        new_col_vals = []
    
        for count, ele in enumerate(list(df[col])):
            if count >= baseline_min and count <= limit_idx:
                z_value = zscore(ele, col_mean, col_stdev)
            else:  # if outside limits of zscoring, don't zscore
                z_value = ele
            new_col_vals.append(z_value)

        df[col] = new_col_vals
    return df

def gaussian_smooth(df, sigma: float = 1.5):
    from scipy.ndimage import gaussian_filter1d
    # df = df.iloc[:, 1:]  # omit first col

    return df.apply(gaussian_filter1d, sigma=sigma, axis=0)

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


def main():

    """this is what gives you a straightline"""
    session_root = r"/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis/BetweenMiceAlignmentData"

    list_of_combos_we_care_about = [

            "Block_Trial_Type_Start_Time_(s)",
        ]

    for combo in list_of_combos_we_care_about:

        #for session in sessions:
        files = find_paths(session_root, f"{combo}","all_speeds_renamed.csv")

        for csv in files:

            print(f"CURR CSV: {csv}")
            df: pd.DataFrame
            df = pd.read_csv(csv)

            t = list(df["Time_(s)"]) # keep time
            df = df.T # cols is timepoint
            df = df.iloc[1:, :]  # omit first row


            # Zscore - across trials (timepoint)
            df = custom_standardize_limit_fixed(
                df,
                baseline_min=0,
                baseline_max=299,
                limit_idx=299
            )

            fig, ax = plt.subplots()
            every_nth = 30
            title = f"Z-scored, Savitzky (n={len(df)})"
            xlabel = "Time from trial start (s)"
            ylabel = "Speed (cm/s)"

            # Savgol filter - within trial
            for row_idx in range(len(df)):
                arr_no_nan = list(df.iloc[row_idx,:])
                sav_z_arr = savgol_filter(arr_no_nan, window_length=33, polyorder=2, mode="nearest")
                # change the df
                df.iloc[row_idx,:] = sav_z_arr
                ax.plot(t, sav_z_arr)

            #print(df.head())

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

            plot_path = csv.replace(".csv","_z_-5_5_savgol.png")
            plt.savefig(plot_path)
            plt.close()

            df = df.T
            df.insert(0, "Time_(s)", t)

            new_csv_path = csv.replace(".csv","_z_-5_5_savgol.csv")
            df.to_csv(new_csv_path, index=False)


def process_one():

    csv = "/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis/BetweenMiceAlignmentData/BLA_NAcShell/ArchT/Choice/Block_Trial_Type_Start_Time_(s)/(2.0, 'Free')/all_speeds_renamed.csv"

    print(f"CURR CSV: {csv}")
    df: pd.DataFrame
    df = pd.read_csv(csv)

    t = list(df["Time_(s)"]) # keep time
    df = df.T # cols is timepoint
    df = df.iloc[1:, :]  # omit first row


    # Zscore
    df = custom_standardize_limit_fixed(
        df,
        baseline_min=0,
        baseline_max=299,
        limit_idx=299
    )

    fig, ax = plt.subplots()
    every_nth = 30
    title = f"Z-scored, Savitzky (n={len(df)})"
    xlabel = "Time from trial start (s)"
    ylabel = "Speed (cm/s)"

    # Savgol filter
    for row_idx in range(len(df)):
        arr_no_nan = list(df.iloc[row_idx,:])
        sav_z_arr = savgol_filter(arr_no_nan, window_length=33, polyorder=2)
        # change the df
        df.iloc[row_idx,:] = sav_z_arr
        ax.plot(t, sav_z_arr)

    #print(df.head())

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

    plot_path = csv.replace(".csv","_z_-5_5_savgol.png")
    plt.savefig(plot_path)
    plt.close()

    df = df.T
    df.insert(0, "Time_(s)", t)

    new_csv_path = csv.replace(".csv","_z_-5_5_savgol.csv")
    df.to_csv(new_csv_path, index=False)


if __name__ == "__main__":
    main()
    #process_one()