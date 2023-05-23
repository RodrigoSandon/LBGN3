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

def convert_min_to_sec(min, sec):
    min_to_sec = int(min) * 60
    total_sec = min_to_sec + float(sec)
    return total_sec

def convert_sec_to_frames(sec, fps):
    frame_num = float(sec) * fps
    return frame_num

def break_down_timestamp(timestamp):
    """_summary_

    Args:
        timestamp (str): _description_

    Returns:
        int: _description_
        int: _description_
    """
    min = int(timestamp.split(":")[0])
    sec = int(timestamp.split(":")[1])
    return min, sec

def create_timestamps(start_timestamp: str, num_events: int, step: int, fps: float, units_steps, units_timestamp):
    """_summary_

    Args:
        start_timestamp (str): when to start the timestamps
        num_events (int): number of events
        step (int): step size, how many frames to skip
        fps (float): frames per second of video
        units_steps (str): the units of the step size
        units_timestamp (str): the units of the start timestamp

    Returns:
        list: list of timestamps
    """
    if units_timestamp == "min":
        min, sec = break_down_timestamp(start_timestamp)
        start_frame_num = convert_sec_to_frames(convert_min_to_sec(min, sec), fps)
    elif units_timestamp == "sec":
        start_frame_num = convert_sec_to_frames(start_timestamp, fps)

    if units_steps == "sec":
        step_in_frames  = convert_sec_to_frames(step, fps)
    elif units_steps == "min":
        min, sec = break_down_timestamp(step)
        step_in_frames = convert_sec_to_frames(convert_min_to_sec(min, sec), fps)   

    #print("start_frame_num: ", start_frame_num)
    #print("step_in_frames: ", step_in_frames)
    frame_stamps = [start_frame_num + i * step_in_frames for i in range(0, num_events)]
    #print("frame_stamps: ", frame_stamps)
    return frame_stamps


def timing_file_processing(file_path, fps, start_time_in_frames):
    df = pd.read_csv(file_path)
    #iterate through columns except trial column
    for col in list(df.columns):
        #print(list(df[col]))
        if col != "Trial":
            # determine if it's in min:sec format or sec format
            if ":" in str(list(df[col])[0]):
                time_format_to_frames(df, col, fps, start_time_in_frames)
            else:
                secs_to_frames(df, col, fps, start_time_in_frames)

    print(df.head())
    return df

def time_format_to_frames(df: pd.DataFrame, col, fps, start_time_in_frames):
    new_lst = []
    old_lst = list(df[col])
    for i in old_lst:
        min = int(i.split(":")[0])
        sec = int(i.split(":")[1])
        min_to_sec = min * 60
        total_sec = min_to_sec + sec
        frame_num = (total_sec * fps) + start_time_in_frames
        new_lst.append(frame_num)
    df[col] = new_lst

def secs_to_frames(df: pd.DataFrame, col, fps, start_time_in_frames):
    new_lst = []
    old_lst = list(df[col])
    for i in old_lst:
        frame_num = (i * fps) + start_time_in_frames
        new_lst.append(frame_num)
    df[col] = new_lst


def freezing_output_processing(file_path):
    df_result = pd.read_csv(file_path)

    frame_list = list(df_result["Frame"])
    frame_list = [i + count for count, i in enumerate(frame_list)]
    df_result["Frame"] = frame_list

    #print(df_result.head())
    return df_result

def replace_func(x, col, timestamps):
    if x in timestamps:
        print(col)
        return col
    else:
        return x

# do this after their processing
def freezing_alignment(df_freezing_out: pd.DataFrame, df_timing: pd.DataFrame):
    # add empty column first (zero-filled)
    df_freezing_out["Timestamps"] = [0] * len(df_freezing_out)
    replace_lst = list(df_freezing_out["Timestamps"])
    #print("here")

    for col in list(df_timing.columns):
            if col != "Trial":
                timestamps = list(df_timing[col])
                print("timestamps")
                print(timestamps)

                #replace_func = lambda x: col if x in timestamps else x
                new_series = df_freezing_out["Frame"].apply(lambda x: replace_func(x, col, timestamps)).tolist()
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

            # these could be floats
            lower_bound_idx = int(idx)
            upper_bound_idx = int(idx + (half_time_window * fps))

            time_str = f"{val}:{time} : {frame_lst[idx]}"
            #print(time_str)
            binned_timestamps_lst.append(time_str)
            #print(lower_bound_idx, upper_bound_idx)

            freezing_sublst = freezing_lst[lower_bound_idx : upper_bound_idx]
            freezing_proportion = get_proportion_freezing(freezing_sublst)
            binned_freezing_lst.append(freezing_proportion)

    return binned_timestamps_lst, binned_freezing_lst
