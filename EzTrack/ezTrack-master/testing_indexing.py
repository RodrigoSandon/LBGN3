import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import timedelta
import os
import math
import random
import matplotlib.colors as mcolors


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
    print(freezing_sublst)
    print("len", len(freezing_sublst))
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
            print("seconds:", (frame_lst[idx] / fps))

            # now go back and forth 30 secs (900 frames)
            # only focues on time periods after CS ON
            #lower_bound_idx = idx - (half_time_window * fps)
            lower_bound_idx = idx
            upper_bound_idx = idx + (half_time_window * fps)

            time_str = f"{val}:{time} : {frame_lst[idx]}"
            print(time_str)
            binned_timestamps_lst.append(time_str)

            freezing_sublst = freezing_lst[lower_bound_idx : upper_bound_idx]
            freezing_proportion = get_proportion_freezing(freezing_sublst)
            binned_freezing_lst.append(freezing_proportion)

    #print(binned_timestamps_lst)
    #print(len(binned_timestamps_lst))
    #print(binned_freezing_lst)
    #print(len(binned_freezing_lst))
    return binned_timestamps_lst, binned_freezing_lst



def main():
    experiment_type = "Conditioning"

    opsin_group_colors = [mcolors.to_hex((random.random(), random.random(), random.random())), 
                          mcolors.to_hex((random.random(), random.random(), random.random()))]

    event_tracked = 'CS ON'
    half_time_window = 30
    fps = 30
    
    ROOT = f"/media/rory/Padlock_DT/Fear_Conditioning_Control/NewVideos/{experiment_type}"
    file = "/media/rory/Padlock_DT/Fear_Conditioning_Control/NewVideos/Conditioning/RRD276_Conditioning_FreezingOutput_processed.csv"

    ROOT_TIMING_FILE = "/media/rory/Padlock_DT/Fear_Conditioning_Control/"
    timing_file_name = f"{experiment_type}_CS_timing_FC_Control.csv"
    timing_filepath = os.path.join(ROOT_TIMING_FILE, timing_file_name)
    df_timing = pd.read_csv(timing_filepath)
    cs_nums = range(1, len(df_timing) + 1)

    experimental_groups_csv = "/media/rory/Padlock_DT/Fear_Conditioning_Control/experimental_groups.csv"
    experimental_groups_df = pd.read_csv(experimental_groups_csv)

    fig, ax = plt.subplots()

    grouped_df = experimental_groups_df.groupby("opsin").agg(lambda x: list(x))
    d_from_df = grouped_df.T.to_dict(orient='list')
    #print(d_from_df)

    # flatten 2d array
    for key, values in d_from_df.items():
        #print(key)
        d_from_df[key] = values[0]
        #print(d_from_df[key])

    # new d will be created of avgs of groups
    d_groups = {}   

    num_mice = 0
  
    mouse_num = 276
    opsin = None

    # check if mouse is in one of the opsin groups
    for key, values in d_from_df.items():
        if mouse_num in d_from_df[key]:
            opsin = key
            
    print(mouse_num, ":", opsin)

    processed_freezing_out_filename = file
    processed_freezing_out_path = os.path.join(ROOT, processed_freezing_out_filename)
    
    df = pd.read_csv(processed_freezing_out_path)
    frame_lst = list(df["Frame"])

    timestamps_lst = list(df["Timestamps"])
    
    # stamped_lst is the x
    stamped_lst = overlap_two_lists(frame_lst, timestamps_lst)

    # modify y to just be binary and not 0 and 100
    freezing_lst = lst_to_binary_lst(list(df["Freezing"]))
    #print(freezing_lst)

    # half_time_window is in seconds
    x, proportions = bin_data(frame_lst, timestamps_lst,freezing_lst, half_time_window = half_time_window, fps=fps, event_tracked=event_tracked)
    #list_of_freezing_props_all_mice.append(proportions)

    # add to d
    if opsin in d_groups:
        d_groups[opsin].append(proportions)
    else:
        d_groups[opsin] = []
        d_groups[opsin].append(proportions)

    num_mice += 1
        
    count = 0
    for key in d_groups:

        # Convert the list of lists to a NumPy array
        array_of_lists = np.array(d_groups[key])

        # Calculate the average of the array along the columns (axis=0)
        average = np.mean(array_of_lists, axis=0)
        #print(average)

        # Calculate the standard deviation of the array along the columns (axis=0)
        std_deviation = np.std(array_of_lists, axis=0)
        #print(std_deviation)
        std_error = [std / math.sqrt(len(average)) for std in std_deviation]

        ax.plot(cs_nums, average, label=key, color=opsin_group_colors[count])
        #print(hex_color)
        plt.errorbar(cs_nums, average, yerr = std_error, fmt='-o', color=opsin_group_colors[count], capsize=3)

        outfilename = f"{experiment_type}_halftimewdw{half_time_window}_fps{fps}_plot.png"
        outpath = os.path.join(ROOT, outfilename)
        count += 1

    ax.set_title(f"Proportion of Freezing - {experiment_type} (n={num_mice})")
    ax.set_ylabel(f"Proportion")
    ax.set_xlabel(f"CS #")
    plt.legend()
    print(outpath)
    fig.savefig(outpath)
    #plt.show()
    plt.close()


if __name__ == "__main__":
    main()