import pandas as pd
import Utilities
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


def avg_cell_eventrace(df, csv_path, cell_name, plot: bool, export_avg: bool):
    """Plots the figure from the csv file given"""
    path_to_save = csv_path.replace(
        "plot_ready.csv", "avg_plot_z_pre.png")
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
            "plot_ready.csv", "avg_plot_ready_z_pre.csv")
        export_avg_cell_eventraces(cell_name, avg_of_col_lst, path_to_save)


def export_avg_cell_eventraces(
    cell_name, avg_dff_list_for_timewindow_n_event, out_path
):
    df = pd.DataFrame(avg_dff_list_for_timewindow_n_event, columns=[cell_name])
    df.to_csv(out_path, index=False)


csv_path_3_16 = "/Volumes/T7Touch/NIHBehavioralNeuroscience/misc/plot_ready.csv"

df: pd.DataFrame
df = pd.read_csv(csv_path_3_16)
# print(df.head())
col_to_save = list(df["Event #"])  # save col that u will omit once transposed
df = df.T
df = df.iloc[1:, :]  # omit first row

# print(df.head())

# 1) Zscore
df = Utilities.custom_standardize(
    df,
    unknown_time_min=-10.0,
    unknown_time_max=-5.0,
    reference_pair={0: 100},
    hertz=10,
)
df = df.T
df = Utilities.gaussian_smooth(df)

# 2) Average Z score per each trial
avg_cell_eventrace(
    df, csv_path_3_16, "3_C16", plot=True, export_avg=True
)

df.insert(0, "Event #", col_to_save)

csv_moded_out_path = csv_path_3_16.replace(".csv", "_z_pre.csv")
df.to_csv(csv_moded_out_path, index=False)
