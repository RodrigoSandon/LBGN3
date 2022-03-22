import pandas as pd
import Utilities
import matplotlib.pyplot as plt
import os
from scipy import stats


def export_avg_cell_eventraces(
    cell_name, avg_dff_list_for_timewindow_n_event, out_path
):
    df = pd.DataFrame(avg_dff_list_for_timewindow_n_event, columns=[cell_name])
    df.to_csv(out_path, index=False)


def avg_cell_eventrace(df, col_to_save, csv_path, cell_name, plot: bool, export_avg: bool):
    """Plots the figure from the csv file given"""
    path_to_save = csv_path.replace("plot_ready.csv", "avg_plot_z.png")
    #df_sub = df.iloc[:, 1:]
    # print(df_sub.head())
    xaxis = list(df.columns)

    row_count = len(df)

    avg_of_col_lst = []
    for col_name, col_data in df.iteritems():
        avg_dff_of_timewindow_of_event = df[col_name].mean()
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
            "plot_ready.csv", "avg_plot_ready_z.csv")
        export_avg_cell_eventraces(cell_name, avg_of_col_lst, path_to_save)


def custom_standardize(
    df: pd.DataFrame, unknown_time_min, unknown_time_max, reference_pair: dict, hertz: int
):

    for col in df.columns:
        subwindow = create_subwindow_for_col(
            df, col, unknown_time_min, unknown_time_max, reference_pair, hertz
        )
        mean_for_cell = stats.tmean(subwindow)
        stdev_for_cell = stats.tstd(subwindow)
        #print(f"mean for {col}: {mean_for_cell}\n")
        #print(f"stdev for {col}: {stdev_for_cell}\n")

        new_col_vals = []
        for ele in list(df[col]):
            #print(f"value: {ele}")
            z_value = zscore(ele, mean_for_cell, stdev_for_cell)
            #print(f"z: {z_value}")
            new_col_vals.append(z_value)
        # print(new_col_vals[0:10])  # has nan values
        df[col] = new_col_vals  # <- not neccesary bc of the .apply function?
    return df


def zscore(obs_value, mu, sigma):
    return (obs_value - mu) / sigma


def convert_secs_to_idx(
    unknown_time_min, unknown_time_max, reference_pair: dict, hertz: int
):
    reference_time = list(reference_pair.keys())[0]  # has to come from 0
    reference_idx = list(reference_pair.values())[0]
    idx_start = (unknown_time_min * hertz) + reference_idx

    idx_end = (unknown_time_max * hertz) + reference_idx
    return int(idx_start), int(idx_end)


def create_subwindow_for_col(
    df, col, unknown_time_min, unknown_time_max, reference_pair, hertz
) -> list:
    idx_start, idx_end = convert_secs_to_idx(
        unknown_time_min, unknown_time_max, reference_pair, hertz
    )
    # print(idx_start, idx_end)
    subwindow = df[col][idx_start:idx_end]
    # print(subwindow)
    return subwindow


csv_path = "/Users/rodrigosandon/Downloads/plot_ready.csv"
df = pd.read_csv(csv_path)
col_to_save = list(df["Event #"])
df = df.T # 1 col is 1 trial currently
df = df.iloc[1:, :]  # omit first row
print(df.head())

# 1) Zscore
df = custom_standardize(
    df,
    unknown_time_min=-10.0,
    unknown_time_max=0.0,
    reference_pair={0: 100},
    hertz=10,
)
df = df.T
# guassian runs on x-axis (rows) (curently 1 col is 1 timepoint currently)
df = Utilities.gaussian_smooth(df)

# 2) Average Z score per each trial
# this will avg across number of columns (across each timepoint)
avg_cell_eventrace(
    df, col_to_save, csv_path, cell_name="C01", plot=True, export_avg=True
)
