import Utilities
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


def avg_cell_eventrace(df, csv_path, cell_name, plot: bool, export_avg: bool):
    """Plots the figure from the csv file given"""
    path_to_save = csv_path.replace(
        "plot_ready.csv", "avg_plot_z_pre_3_16.png")
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
            "plot_ready.csv", "avg_plot_ready_z_pre_3_16.csv")
        export_avg_cell_eventraces(cell_name, avg_of_col_lst, path_to_save)


def export_avg_cell_eventraces(
    cell_name, avg_dff_list_for_timewindow_n_event, out_path
):
    df = pd.DataFrame(avg_dff_list_for_timewindow_n_event, columns=[cell_name])
    df.to_csv(out_path, index=False)

def zscore(obs_value, mu, sigma):
    return (obs_value - mu) / sigma

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

csv_path_3_16 = "/media/rory/Padlock_DT/BLA_Analysis/Debugging/high_z_score_issue/plot_ready.csv"

df: pd.DataFrame
df = pd.read_csv(csv_path_3_16)
# print(df.head())
col_to_save = list(df["Event #"])  # save col that u will omit once transposed
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
    df, csv_path_3_16, "3_C16", plot=True, export_avg=True
)

df.insert(0, "Event #", col_to_save)

csv_moded_out_path = csv_path_3_16.replace(".csv", "_z_pre_3_16.csv")
df.to_csv(csv_moded_out_path, index=False)

################################################################
