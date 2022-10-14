import os, glob
from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import statistics


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

def make_avg_speed_table(filename, csv_path_speed, half_of_time_window, fps):

    df_speed = pd.read_csv(csv_path_speed)

    #df_speed.columns = df_speed.iloc[0]
    #df_speed = df_speed.iloc[1:, :]
    #omitting first column
    df_speed = df_speed.iloc[:, 1:]
    #print(df_speed.head())

    seconds_in_frame = float(1/fps)
    # 0.03333 is due to the 30Hz
    x_axis = np.arange(-half_of_time_window, half_of_time_window, seconds_in_frame).tolist()
    #print(x_axis)
    #x_axis = [round(i, 1) for i in x_axis]
    x_axis = x_axis[:-1]
    
    avg_of_col_speed_lst = []
    for col in list(df_speed.columns):
        timepoint_avg = statistics.mean(list(df_speed[col]))
        avg_of_col_speed_lst.append(timepoint_avg)

    #print("here:", len(x_axis), len(avg_of_col_speed_lst))
    #print(len(x_axis))
    #print(len(avg_of_col_speed_lst[:-1]))
    csv_prep_unnorm = {
        "Time_(s)" : x_axis,
        "Avg_Speed_(cm/s)" : avg_of_col_speed_lst[:-1]
    }

    path_to_save = csv_path_speed.replace(filename, "avg_speed.csv")
    dff_n_speed = pd.DataFrame.from_dict(csv_prep_unnorm)
    dff_n_speed.to_csv(path_to_save, index = False)

    return path_to_save

def plot_avg_speed(csv_path, event_num, fps):
    """Plots the figure from the csv file given"""
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots()
    every_nth = fps
    ax.plot(list(df["Time_(s)"]), list(df["Avg_Speed_(cm/s)"]))
    ax.set_xticks([round(i, 1) for i in list(df["Time_(s)"])])
    ax.set_xlabel("Time from trigger (s)")
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    for n, label in enumerate(ax.xaxis.get_major_ticks()):
        if n % every_nth != 0:
            label.set_visible(False)
    ax.set_ylabel("Average speed (cm/s)")
    ax.set_title(f"Avg. Speed of Event (n={event_num})")
    fig.savefig(csv_path.replace(".csv",".png"))
    plt.close(fig)

def plot_indv_speeds(csv_path, filename, fps):

    df = pd.read_csv(csv_path)
    df = df.iloc[:, 1:]

    new_path = csv_path.replace(filename, "speeds.png")
    df = pd.read_csv(csv_path)
    number_of_events = df.shape[0]
    
    df_without_eventcol = df.loc[:, df.columns != "Event_#"]
    
    just_event_col = df.loc[:, df.columns == "Event_#"]

    df = pd.concat([just_event_col, df_without_eventcol], axis=1)
    df = df.T

    new_header = df.iloc[0]  # first row
    df = df[1:]  # don't include first row in new df
    df.columns = new_header
    

    x = list(df.index)
    """t_pos = np.arange(0.00, 5.03, 0.03)
    t_neg = np.arange(-5.00, 0.00, 0.03)
    t = t_neg.tolist() +  t_pos.tolist()
    t = [round(i, 1) for i in t]"""

    fig, ax = plt.subplots()
    every_nth = fps

    for col in df.columns:

        if col != "Event_#":         
            ax.plot(x, list(df[col]))

    ax.set_xticks(x)
    ax.set_xlabel("Time from trigger (s)")
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    for n, label in enumerate(ax.xaxis.get_major_ticks()):
        if n % every_nth != 0:
            label.set_visible(False)
    ax.set_ylabel("Speed (cm/s)")
    ax.set_title(f"Speed of events (n={number_of_events})")
    fig.savefig(new_path)
    plt.close(fig)

def export_avg_cell_eventraces(
    cell_name, avg_dff_list_for_timewindow_n_event, out_path
):
    df = pd.DataFrame(avg_dff_list_for_timewindow_n_event, columns=[cell_name])
    df.to_csv(out_path, index=False)


def find_paths_endswith(session_path, endswith):
    files = glob.glob(
        os.path.join(session_path, "**", "*%s") % (endswith),
        recursive=True,
    )
    return files


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
    return ["%s_%s" % (basename, count + 1) for count in range(desired_rows)]


def get_colnames_aslist(df):
    return [col for col in df.columns]


def rename_all_col_names(df, newnames_list):
    col_rename_dict = {i: j for i, j in zip(get_colnames_aslist(df), newnames_list)}
    df = df.rename(columns=col_rename_dict)
    return df
