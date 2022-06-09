import os
import glob
from typing import List
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
from scipy import signal
from scipy import stats
import numpy as np


def create_combos(event_name_list_input: List):
    number_items_to_select = list(range(event_name_list_input + 1))
    for i in number_items_to_select:
        to_select = i
        combs = combinations(event_name_list_input, to_select)


def avg_cell_eventrace(df, csv_path, cell_name, plot: bool, export_avg: bool):
    """Plots the figure from the csv file given"""
    path_to_save = csv_path.replace("plot_ready_choice_aligned.csv", "avg_plot_choice_aligned.png")
    #df_sub = df.iloc[:, 1:]
    # print(df_sub.head())
    xaxis = list(df.columns)
    xaxis_len = len(xaxis)

    row_count = len(df)

    # averages columns (timepoints) while skipping NaNs
    avg_of_col_lst = list(df.mean(skipna=True))

    if plot == True:
        fig, ax = plt.subplots()
        every_nth = 30
        ax.plot(xaxis, avg_of_col_lst)
        ax.set_title(("Average DF/F Trace for %s Event Window") % (cell_name))
        ax.set_xlabel("Time relative to choice (s)")
        for n, label in enumerate(ax.xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)
        for n, label in enumerate(ax.xaxis.get_major_ticks()):
            if n % every_nth != 0:
                label.set_visible(False)
        ax.set_ylabel("Average DF/F (n=%s)" % (row_count))
        fig.savefig(path_to_save)
        plt.close(fig)

    if export_avg == True:
        path_to_save = csv_path.replace(
            "plot_ready_choice_aligned.csv", "avg_plot_ready_choice_aligned.csv")
        export_avg_cell_eventraces(cell_name, avg_of_col_lst, path_to_save)

def avg_cell_eventrace_w_resampling(df: pd.DataFrame, csv_path, cell_name, x_ticks, plot: bool, export_avg: bool):
    """Plots the figure from the csv file given"""
    path_to_save = csv_path.replace("plot_ready_choice_aligned_resampled.csv", "avg_plot_choice_aligned_resampled.png")
    xaxis = list(df.columns)
    xaxis_len = len(xaxis)

    row_count = len(df)

    # averages columns (timepoints) while skipping NaNs
    avg_of_col_lst = list(df.mean(skipna=True))

    if plot == True:
        fig, ax = plt.subplots()
        every_nth = 20
        ax.plot(xaxis, avg_of_col_lst)
        #ax.set_xticks(x_ticks)
        ax.set_xlabel("Time relative to choice (s)")
        for n, label in enumerate(ax.xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)
        for n, label in enumerate(ax.xaxis.get_major_ticks()):
            if n % every_nth != 0:
                label.set_visible(False)
        ax.set_ylabel("Resampled Average DF/F (n=%s)" % (row_count))
        fig.savefig(path_to_save)
        plt.close(fig)

    if export_avg == True:
        path_to_save = csv_path.replace(
            "plot_ready_choice_aligned_resampled.csv", "avg_plot_ready_choice_aligned_resampled.csv")
        export_avg_cell_eventraces(cell_name, avg_of_col_lst, path_to_save)


def create_resampled_plot_ready(csv_path, t, desired_start_choice_len, desired_choice_end_len):
    df = pd.read_csv(csv_path)
    col_to_save = list(df.iloc[:, 0])
    #print(col_to_save)
    df = df.iloc[: , 1:]
    #print(df.head())
    df = df.T

    d = {}

    # temporarily have col represent all trials so u can resample them
    for col in list(df.columns):
    
        trial_start_to_choice = [i for i in df.iloc[:df.index.get_loc("Choice Time"), col] if pd.isna(i) == False]
        trial_choice_to_end = [i for i in df.iloc[df.index.get_loc("Choice Time") + 1:, col] if pd.isna(i) == False]

        resampled_start_to_choice = signal.resample(trial_start_to_choice, desired_start_choice_len)
        resampled_choice_to_end = signal.resample(trial_choice_to_end, desired_choice_end_len)

        final_resampled = list(resampled_start_to_choice) + [df.iloc[df.index.get_loc("Choice Time"), col]] + list(resampled_choice_to_end)
        
        d[col] = final_resampled

    new_df = pd.DataFrame.from_dict(d)
    new_df = new_df.T
    new_df.columns = t
    new_df.insert(0,"Event #", col_to_save)
    new_csv_path = csv_path.replace("plot_ready_choice_aligned.csv", "plot_ready_choice_aligned_resampled.csv")
    new_df.to_csv(new_csv_path, index=False)



def export_avg_cell_eventraces(
    cell_name, avg_dff_list_for_timewindow_n_event, out_path
):
    df = pd.DataFrame(avg_dff_list_for_timewindow_n_event, columns=[cell_name])
    df.to_csv(out_path, index=False)


def find_dff_trace_path(session_path, endswith):
    files = glob.glob(
        os.path.join(session_path, "**", "*%s") % (endswith),
        recursive=True,
    )
    if len(files) == 1:  # should only find one
        print(f"FILE FOUND: , {files[0]}")
        return files[0]
    elif len(files) == 0:
        return None
    else:
        print(("More than one file ending with %s found!") % (endswith))

    """Will take in the neuron categorized dff traces and the abet data.
    This should be in the EventTraces class.
    """


"""TODO: Add another parameter to this function, the dlc data, eventually"""


def binary_search(data, val):
    """Will return index if the value is found, otherwise the index of the item that is closest
    to that value."""
    lo, hi = 0, len(data) - 1
    best_ind = lo
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if data[mid] < val:
            lo = mid + 1
        elif data[mid] > val:
            hi = mid - 1
        else:
            best_ind = mid
            break
        # check if data[mid] is closer to val than data[best_ind]
        if abs(data[mid] - val) < abs(data[best_ind] - val):
            best_ind = mid

    return best_ind


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


def insert_time_index_to_df(df: pd.DataFrame, range_min, range_max, step) -> pd.DataFrame:
    x_axis = np.arange(range_min, range_max, step).tolist()
    # end shoudl be 10.1 and not 10 bc upper limit is exclusive

    middle_idx = int(len(x_axis) / 2)

    end_idx = len(x_axis) - 1
    # print(x_axis[end_idx])

    #x_axis[middle_idx] = 0
    x_axis = [round(i, 1) for i in x_axis]

    df.insert(0, "Time (s)", x_axis)
    df = df.set_index("Time (s)")

    return df


def zscore(obs_value, mu, sigma):
    return (obs_value - mu) / sigma


def convert_secs_to_idx(
    unknown_time_min, unknown_time_max, reference_pair: dict, hertz: int
):
    reference_time = list(reference_pair.keys())[0]  # has to come from 0
    reference_idx = list(reference_pair.values())[0]

    # first find the time difference between reference and unknown
    # Note: reference will
    idx_start = (unknown_time_min * hertz) + reference_idx
    # idx_end = (unknown_time_max * hertz) + reference_idx + 1
    # ^plus 1 bc getting sublist is exclusive? 11/30/21
    idx_end = (unknown_time_max * hertz) + reference_idx
    return int(idx_start), int(idx_end)


def custom_standardize(
    df: pd.DataFrame, unknown_time_min, unknown_time_max, reference_pair: dict, hertz: int
):
    # df = df.iloc[:, 1:]  # omit first col

    # print(df.head())
    for col in df.columns:
        subwindow = create_subwindow_for_col(
            df, col, unknown_time_min, unknown_time_max, reference_pair, hertz
        )
        mean_for_cell = stats.tmean(subwindow)
        stdev_for_cell = stats.tstd(subwindow)
        # print(subwindow)
        # print(f"Mean {mean_for_cell} for cell {col}")
        # print(stdev_for_cell)

        new_col_vals = []
        for ele in list(df[col]):
            z_value = zscore(ele, mean_for_cell, stdev_for_cell)
            new_col_vals.append(z_value)

        # print(new_col_vals[0:10])  # has nan values
        df[col] = new_col_vals  # <- not neccesary bc of the .apply function?
    return df


def gaussian_smooth(df, sigma: float = 1.5):
    from scipy.ndimage import gaussian_filter1d
    # df = df.iloc[:, 1:]  # omit first col

    return df.apply(gaussian_filter1d, sigma=sigma, axis=0)


def make_value_list_for_col(basename, desired_rows):
    return ["%s %s" % (basename, count + 1) for count in range(desired_rows)]


def get_colnames_aslist(df):
    return [col for col in df.columns]


def rename_all_col_names(df, newnames_list):
    col_rename_dict = {i: j for i, j in zip(
        get_colnames_aslist(df), newnames_list)}
    df = df.rename(columns=col_rename_dict)
    return df