import cv2
import os, glob
import pandas as pd
import numpy as np
import holoviews as hv
import FreezeAnalysis_Functions as fz
import EzTrackFunctions as ez
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import timedelta
import math
import random
import matplotlib.colors as mcolors

def main():
    people = ["Patrick", "Olena", "Ozge"]
    eztrack_output_processed_suffix = "FreezingOutput_processed.csv"
    ROOT = f"/media/rory/Padlock_DT/Fear_Conditioning_Control/Olena/"

    root_calibration_vids = f"/media/rory/Padlock_DT/Fear_Conditioning_Control/NewVideos/Calibration"

    experiment_type = "conditioning"

    
    timing_file_name = f"olena_FC_info_{experiment_type}.csv"

    FreezeThresh = 180 
    MinDuration = 40
    number_of_frames_to_calibrate = 600
    calibrate_video_what_frame_to_start = 0
    dsmpl = 1
    h,w = 300,1000  
    vid_d_start = 0
    event_tracked = 'CS ON'
    half_time_window = 30
    fps = 30

    # so ima builed this so it doesn't have to read the csv file every time, just have your desired time stamps for each event
    # I need starting step, frames, and how many steps to do, and current unit in which i have info of
    
    timing_filepath = os.path.join(ROOT_TIMING_FILE, timing_file_name)
    ROOT = os.path.join(ROOT, experiment_type)

    df_correspondence = pd.read_csv(correspondence_filepath)
    print(df_correspondence.head())

    vid_list = list(df_correspondence[colname_vid_paths])

    vid = vid_list[0]

    print(f"vid: {vid}")

    vid_name = vid.split("/")[-1]

    chamber = df_correspondence.query(f"{colname_vid_paths}=='{vid}'")[letter_column_name]
    chamber = chamber.values[0]

    # the calibration video must contain the two: experiment type and chamber
    if experiment_type == "Retrieval":
        calibration_vid_file = f"Chamber_{chamber}_calibration_extinction.avi"
    else:
        calibration_vid_file = f"Chamber_{chamber}_calibration_{experiment_type.lower()}.avi"

    cal_dif_avg: float # Average frame-by-frame pixel difference
    percentile: float # 99.99 percentile of pixel change differences
    mt_cutoff: float # Grayscale change cut-off for pixel change

    video_dict = {
        'dpath'   : root_calibration_vids,  
        'file'    : calibration_vid_file,
        'start'   : calibrate_video_what_frame_to_start, 
        'end'     : None,
        'dsmpl'   : dsmpl,
        'stretch' : dict(width=1, height=1),
        'cal_frms' : number_of_frames_to_calibrate
        }


    img_crp, video_dict = fz.LoadAndCrop(video_dict)

    ####### CALIBRATION #######
    cal_dif_avg, percentile, mt_cutoff = fz.calibrate_custom(video_dict, cal_pix=10000, SIGMA=1)

    ####### FREEZE ANALYSIS #######
    cap = cv2.VideoCapture(vid)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("length of vid: ", length)

    vid_d_end = length

    video_dict = {
        'dpath'   : ROOT,  
        'file'    : vid_name,
        'fpath'   : vid,
        'start'   : vid_d_start, 
        'end'     : vid_d_end,
        'dsmpl'   : dsmpl,
        'stretch' : dict(width=1, height=1)
    }

    Motion, frames_processed = fz.Measure_Motion(video_dict, mt_cutoff, SIGMA=1)  
    plt_mt = hv.Curve((np.arange(len(Motion)),Motion),'Frame','Pixel Change').opts(
        height=h,width=w,line_width=1,color="steelblue",title="Motion Across Session")
    plt_mt

    Freezing = fz.Measure_Freezing(Motion,FreezeThresh,MinDuration)  
    fz.SaveData(video_dict,Motion,Freezing,mt_cutoff,FreezeThresh,MinDuration)
    print('Average Freezing: {x}%'.format(x=np.average(Freezing)))

    vid_name_no_ext = vid_name.split(".")[0]
    result_filename = f"{vid_name_no_ext}_FreezingOutput.csv"
    result_path = os.path.join(ROOT, result_filename)

    while True:
        user_input = input("Do you want to continue running the next part of the code (ezTrack output processing)? (y/n): ")
        if user_input.lower() == "y":
            # Code to continue running goes here

            for file in os.listdir(ROOT):
                if "FreezingOutput.csv" in file:
                    file_path = os.path.join(ROOT, file)
                    file_out_path = file_path.replace(".csv", "_processed.csv")
                    print(file_path)

                    df_freezing_out = pd.read_csv(file_path)
                    df_timing = ez.timing_file_processing(timing_filepath, fps)
                    df_aligned = ez.freezing_alignment(df_freezing_out, df_timing)

                    df_aligned.to_csv(file_out_path, index=False)
            break
        
        elif user_input.lower() == "n":
            print("Exiting program...")
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    while True:
        user_input = input("Do you want to continue running the next part of the code (binning and plotting)? (y/n): ")
        if user_input.lower() == "y":
            # Code to continue running goes here
            opsin_group_colors = [mcolors.to_hex((random.random(), random.random(), random.random())), 
                            mcolors.to_hex((random.random(), random.random(), random.random()))]

            df_timing = pd.read_csv(timing_filepath)
            cs_nums = range(1, len(df_timing) + 1)

            fig, ax = plt.subplots()

            grouped_df = experimental_groups_df.groupby("opsin").agg(lambda x: list(x))
            d_from_df = grouped_df.T.to_dict(orient='list')

            # flatten 2d array
            for key, values in d_from_df.items():
                d_from_df[key] = values[0]

            # new d will be created of avgs of groups
            d_groups = {}   

            num_mice = 0
            for file in os.listdir(ROOT):
                if eztrack_output_processed_suffix in file:
                    mouse_num = int(file.split("_")[0].replace("RRD", ""))
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
                    stamped_lst = ez.overlap_two_lists(frame_lst, timestamps_lst)

                    # modify y to just be binary and not 0 and 100
                    freezing_lst = ez.lst_to_binary_lst(list(df["Freezing"]))
                    #print(freezing_lst)

                    # half_time_window is in seconds
                    x, proportions = ez.bin_data(frame_lst, timestamps_lst,freezing_lst, half_time_window = half_time_window, fps=fps, event_tracked=event_tracked)
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

                # Calculate the standard deviation of the array along the columns (axis=0)
                std_deviation = np.std(array_of_lists, axis=0)
                std_error = [std / math.sqrt(num_mice) for std in std_deviation]

                ax.plot(cs_nums, average, label=key, color=opsin_group_colors[count])
                plt.errorbar(cs_nums, average, yerr = std_error, fmt='-o', color=opsin_group_colors[count], capsize=3)

                outfilename = f"{experiment_type}_halftimewdw{half_time_window}_fps{fps}_plot.png"
                outpath = os.path.join(ROOT, outfilename)
                count += 1

            ax.set_title(f"Proportion of Freezing - {experiment_type} (n={num_mice})")
            ax.set_ylabel(f"Proportion")
            ax.set_xlabel(f"CS #")
            plt.legend()
            fig.savefig(outpath)
            plt.close()

            break
        elif user_input.lower() == "n":
            print("Exiting program...")
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")



if __name__ == "__main__":
    main()