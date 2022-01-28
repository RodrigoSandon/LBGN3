import os, glob
from typing import List
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations


def create_combos(event_name_list_input: List):
    number_items_to_select = list(range(event_name_list_input + 1))
    for i in number_items_to_select:
        to_select = i
        combs = combinations(event_name_list_input, to_select)


def avg_cell_eventrace(csv_path, cell_name, plot: bool, export_avg: bool):
    """Plots the figure from the csv file given"""
    path_to_save = csv_path.replace("plot_ready.csv", "avg_plot.png")
    df = pd.read_csv(csv_path)
    df_sub = df.iloc[:, 1:]
    # print(df_sub.head())
    xaxis = list(df_sub.columns)

    row_count = len(df_sub)

    avg_of_col_lst = []
    for col_name, col_data in df_sub.iteritems():
        avg_dff_of_timewindow_of_event = df_sub[col_name].mean()
        avg_of_col_lst.append(avg_dff_of_timewindow_of_event)

    if plot == True:

        plt.plot(xaxis, avg_of_col_lst)
        plt.title(("Average DF/F Trace for %s Event Window") % (cell_name))
        plt.xlabel("Time (s)")
        plt.ylabel("Average DF/F (n=%s)" % (row_count))
        plt.savefig(path_to_save)
        plt.close()

    if export_avg == True:
        path_to_save = csv_path.replace("plot_ready.csv", "avg_plot_ready.csv")
        export_avg_cell_eventraces(cell_name, avg_of_col_lst, path_to_save)


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
        print(f"ABET FILE PATH FOR THIS SESSION: , {files[0]}")
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


def make_value_list_for_col(basename, desired_rows):
    return ["%s %s" % (basename, count + 1) for count in range(desired_rows)]


def get_colnames_aslist(df):
    return [col for col in df.columns]


def rename_all_col_names(df, newnames_list):
    col_rename_dict = {i: j for i, j in zip(get_colnames_aslist(df), newnames_list)}
    df = df.rename(columns=col_rename_dict)
    return df
