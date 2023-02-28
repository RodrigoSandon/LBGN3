import cv2
import os, glob
import pandas as pd
import numpy as np
import holoviews as hv
import FreezeAnalysis_Functions as fz
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import timedelta
import math
import random
import matplotlib.colors as mcolors

def timing_file_processing(file_path, fps):
    df = pd.read_csv(file_path)
    #iterate through columns except trial column
    for col in list(df.columns):
        if col != "Trial":
            time_format_to_sec(df, col, fps)

    print(df.head())
    return df

def time_format_to_sec(df: pd.DataFrame, col, fps):
    new_lst = []
    old_lst = list(df[col])
    for i in old_lst:
        min = int(i.split(":")[0])
        sec = int(i.split(":")[1])
        min_to_sec = min * 60
        total_sec = min_to_sec + sec
        frame_num = total_sec * fps
        new_lst.append(frame_num)
    df[col] = new_lst


def freezing_output_processing(file_path):
    df_result = pd.read_csv(file_path)

    frame_list = list(df_result["Frame"])
    frame_list = [i + count for count, i in enumerate(frame_list)]
    df_result["Frame"] = frame_list

    print(df_result.head())
    return df_result

# do this after their processing
def freezing_alignment(df_freezing_out: pd.DataFrame, df_timing: pd.DataFrame):
    # add empty column first (zero-filled)
    df_freezing_out["Timestamps"] = [0] * len(df_freezing_out)
    replace_lst = list(df_freezing_out["Timestamps"])

    for col in list(df_timing.columns):
            if col != "Trial":
                timestamps = list(df_timing[col])

                replace_func = lambda x: col if x in timestamps else x
                new_series = df_freezing_out["Frame"].apply(replace_func).tolist()
                #print(new_series)
                # now replace that old timestamps col with new_series
                for idx, val in enumerate(new_series):
                    if isinstance(val, str):
                        replace_lst[idx] = val

    df_freezing_out["Timestamps"] = replace_lst
    return df_freezing_out

def line_chart(x, y, outpath):
    fig, ax = plt.subplots()

    ax.plot(x, y)

    n = 1000  # every other nth tick
    for i, tick in enumerate(ax.xaxis.get_major_ticks()):
        if i % n != 0:
            tick.label1.set_visible(False)

    plt.show()
    fig.savefig(outpath)

def overlap_two_lists(list1, list2):
    new_lst = []
    for idx, val in enumerate(list2):
        if isinstance(val, str) and val != '0':
            new_lst.append(val)
        else:
            new_lst.append(int(list1[idx]))
    
    return new_lst

def lst_to_binary_lst(lst):
    new_lst = []
    for i in range(len(lst)):
        if lst[i] == 100:
            new_lst.append(1)
        else:
            new_lst.append(0)
    
    return new_lst

def get_proportion_freezing(freezing_sublst):
    num_freezing_points = len([i for i in freezing_sublst if i == 1])
    num_points = len(freezing_sublst)
    proportion = num_freezing_points / num_points

    return proportion

def bin_data(frame_lst, timestamps, freezing_lst, half_time_window, fps, event_tracked):
    binned_timestamps_lst = []
    binned_freezing_lst = []

    for idx, val in enumerate(timestamps):
        # everytime a timestamp is discovered this is what happens
        if val == event_tracked:
            # this means it's a timestamp, get when i happened
            time = timedelta(seconds=(frame_lst[idx] / fps))

            lower_bound_idx = idx
            upper_bound_idx = idx + (half_time_window * fps)

            time_str = f"{val}:{time} : {frame_lst[idx]}"
            print(time_str)
            binned_timestamps_lst.append(time_str)

            freezing_sublst = freezing_lst[lower_bound_idx : upper_bound_idx]
            freezing_proportion = get_proportion_freezing(freezing_sublst)
            binned_freezing_lst.append(freezing_proportion)

    return binned_timestamps_lst, binned_freezing_lst
