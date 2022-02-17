import os, glob
from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
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
    x_axis = np.arange(-10, 10, 0.1).tolist()
    x_axis = [round(i, 1) for i in x_axis]

    row_count = len(df_sub)

    avg_of_col_lst = []
    for col_name, col_data in df_sub.iteritems():
        avg_dff_of_timewindow_of_event = df_sub[col_name].mean()
        avg_of_col_lst.append(avg_dff_of_timewindow_of_event)


    if plot == True:

        plt.plot(x_axis, avg_of_col_lst)
        # set_xticks([i for i in range(0,len(vel_mouse), 600)])
        #plt.xticks([round(i) for i in range(0,len(xaxis), 40)])
        pos_side = [count/20 for count,i in enumerate(xaxis) if count % 20 == 0]
        neg_side = [-(count/20) for count,i in enumerate(xaxis) if count % 20 == 0]
        plt.xticks(neg_side + pos_side)
        plt.title(("Average Df/f Trace for %s Event Window") % (cell_name))
        plt.xlabel("Time (s)")
        plt.ylabel("Average Df/f (n=%s)" % (row_count))
        plt.savefig(path_to_save)
        plt.close()

    if export_avg == True:
        path_to_save = csv_path.replace("plot_ready.csv", "avg_plot_ready.csv")
        export_avg_cell_eventraces(cell_name, avg_of_col_lst, path_to_save)

def zscore(obs_value, mu, sigma):
    return (obs_value - mu) / sigma

def mean_of_lst_o_lsts(df: pd.DataFrame):
    # this df does contain the event # column
    num_of_points_per_timepoint = len(df) - 1

    d = {}
    for col in df:
        d[col] = df[col]

    lists_to_avg = []
    for key in d:
        if key != "Event #":
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


def stdev_of_lst_o_lsts(d: dict):

    lists_to_std = []
    for key in d:
        lists_to_std.append(d[key])
    
    zipped = zip(*lists_to_std)

    std_lst = []

    for index, tuple in enumerate(zipped):
        std_o_timepoint = stats.tstd(list(tuple))
        std_lst.append(std_o_timepoint)
    
    return std_lst

def standardize_indv_traces(df: pd.DataFrame ) -> pd.DataFrame:
    # these dfs do contain the event # column

    standardized_trials = {}
    df = df.T
    #print(df.head())
    time_list = list(df.index)[1:]
    df.columns = df.iloc[0]
    df = df.iloc[1:, :]

    d = {}
    for col in df:
        d[col] = list(df[col])
    
    """for key, value in d.items():
        print(f"{key} : {value}")"""

    mean_of_trials = mean_of_lst_o_lsts(df)
    stdev_of_trials = stdev_of_lst_o_lsts(d)

    # for each trail list of values, we want to convert that to z-scores
    # insert time col first
    standardized_trials["Event #"] = time_list

    for key in d:
        if key != "Event #":
            standardized_trials[key] = []
            for idx, value in enumerate(d[key]):
                z_score_o_timepoint = zscore(value, mean_of_trials[idx], stdev_of_trials[idx])
                standardized_trials[key].append(z_score_o_timepoint)
    
    """for key, value in standardized_trials.items():
        print(f"{key} : {value}")
        break"""

    norm_df = pd.DataFrame.from_dict(standardized_trials)
    norm_df =norm_df.T

    return norm_df

def avg_cell_event_dff_speed_norm(csv_path_dff, csv_path_speed, cell_name, plot: bool, export_avg: bool):
    """Plots the figure from the csv file given"""
    path_to_save = csv_path_dff.replace("plot_ready.csv", "avg_norm_dff_speed_plot.png")
    df_dff = pd.read_csv(csv_path_dff)
    df_speed = pd.read_csv(csv_path_speed)
    """df_dff.columns = df_dff.iloc[1]
    df_speed.columns = df_speed.iloc[1]"""

    df_dff_norm = standardize_indv_traces(df_dff)
    df_dff_norm.columns = df_dff_norm.iloc[0]
    df_dff_norm = df_dff_norm.iloc[1:, :]
    df_speed_norm = standardize_indv_traces(df_speed)
    df_speed_norm.columns = df_speed_norm.iloc[0]
    df_speed_norm = df_speed_norm.iloc[1:, :]

    # Adjust og dffs later, after it is processed in standardized_indv_traces
    df_dff.columns = df_dff.iloc[0]
    df_dff = df_dff.iloc[1:, :]
    df_dff = df_dff.iloc[:, 1:]

    df_speed.columns = df_speed.iloc[0]
    df_speed = df_speed.iloc[1:, :]
    df_speed = df_speed.iloc[:, 1:]
    

    """df_sub_dff = df_dff.iloc[:, 1:]
    df_sub_speed = df_speed.iloc[:, 1:]

    df_sub_dff_norm = df_dff_norm.iloc[:, 1:]
    df_sub_speed_norm = df_speed_norm.iloc[:, 1:]"""
    """print(df_dff_norm.head())
    print(list(df_dff_norm.columns))"""

    xaxis = list(df_dff_norm.columns)
    x_axis = np.arange(-10, 10, 0.1).tolist()
    x_axis = [round(i, 1) for i in x_axis]
    #print(x_axis)
    #print(len(x_axis))
    #x_axis_plot = [i for i in range(-1 * int(abs(float(xaxis[0]))),int(xaxis[-1]), 40)]
    #xaxis_plot = [i.split(".")[0] for count,i in enumerate(xaxis) if count % 20 == 0]

    row_count = len(df_dff_norm)

    avg_of_col_dff_lst = []
    for col_name, col_data in df_dff.iteritems():
        avg_dff_of_timewindow_of_event = df_dff[col_name].mean()
        avg_of_col_dff_lst.append(avg_dff_of_timewindow_of_event)
    
    avg_of_col_speed_lst = []
    for col_name, col_data in df_speed.iteritems():
        avg_dff_of_timewindow_of_event = df_speed[col_name].mean()
        avg_of_col_speed_lst.append(avg_dff_of_timewindow_of_event)
    #print(df_dff_norm.head())
    avg_norm_of_col_dff_lst = []
    for col_name, col_data in df_dff_norm.iteritems():
        avg_dff_of_timewindow_of_event = df_dff_norm[col_name].mean()
        avg_norm_of_col_dff_lst.append(avg_dff_of_timewindow_of_event)
    
    avg_norm_of_col_speed_lst = []
    for col_name, col_data in df_speed_norm.iteritems():
        avg_dff_of_timewindow_of_event = df_speed_norm[col_name].mean()
        avg_norm_of_col_speed_lst.append(avg_dff_of_timewindow_of_event)

    csv_prep_unnorm = {
        cell_name : avg_of_col_dff_lst,
        "Speed (cm/s)" : avg_of_col_speed_lst
    }

    if plot == True:
        
        """print(xaxis)
        print(len(xaxis))
        print(avg_norm_of_col_dff_lst)
        print(len(avg_norm_of_col_dff_lst))"""
        plt.plot(x_axis, avg_norm_of_col_dff_lst, label = "Norm. Df/f")
        plt.plot(x_axis, avg_norm_of_col_speed_lst, label= "Norm. Speed (cm/s)")
        #plt.xticks([round(i) for i in range(0,len(xaxis), 0.1)])
        #print(len([count/20 for count,i in enumerate(xaxis) if count % 20 == 0]))
        pos_side = [count/20 for count,i in enumerate(xaxis) if count % 20 == 0]
        neg_side = [-(count/20) for count,i in enumerate(xaxis) if count % 20 == 0]
        plt.xticks(neg_side + pos_side)
        plt.title(("Average of Z-Scored Df/f Trace for %s Event Window (n=%s)") % (cell_name, row_count))
        plt.xlabel("Time (s)")
        #plt.ylabel("Average Df/f (n=%s)" % (row_count))
        plt.legend()
        plt.savefig(path_to_save)
        plt.close()

    if export_avg == True:
        path_to_save = csv_path_dff.replace("plot_ready.csv", "avg_unnorm_dff_speed_plot_ready.csv")
        dff_n_speed = pd.DataFrame.from_dict(csv_prep_unnorm)
        dff_n_speed.to_csv(path_to_save, index = False)

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


def make_value_list_for_col(basename, desired_rows):
    return ["%s %s" % (basename, count + 1) for count in range(desired_rows)]


def get_colnames_aslist(df):
    return [col for col in df.columns]


def rename_all_col_names(df, newnames_list):
    col_rename_dict = {i: j for i, j in zip(get_colnames_aslist(df), newnames_list)}
    df = df.rename(columns=col_rename_dict)
    return df
