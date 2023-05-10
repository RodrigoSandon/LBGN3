import cv2
import os, glob
import re
import math
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

def find_file_with_strings(root_path, strings_list, eztrack_output_processed_suffix="FreezingOutput_processed.csv"):
    strings_list = [s.lower() for s in strings_list]
    for root, dirs, files in os.walk(root_path):
        for file in files:
            # ignore files that start with a dot (annoying lock files)
            if file.startswith("."):
                continue
            else:
                file_path = os.path.join(root, file).lower()
                if all(string.lower() in file_path for string in strings_list) and eztrack_output_processed_suffix.lower() not in file_path:
                    return os.path.join(root, file)
    return None

def main():

    FreezeThresh = 180 
    MinDuration = 40
    chamber_col = "chamber"
    experimental_group_col = "experimental_group"
    start_time_info_suffix  = "_FC_startime_info.csv"
    eztrack_output_processed_suffix = "FreezingOutput_processed.csv"
    ROOT = "/media/rory/Padlock_DT/Fear_Conditioning_Control/"
    CALIBRATION_ROOT = "/media/rory/Padlock_DT/Fear_Conditioning_Control/calibration"
    calibrate_video_what_frame_to_start = 0
    number_of_frames_to_calibrate = 600
    half_time_window = 2.5
    h,w = 300,1000
    dsmpl = 1
    vid_d_start = 0
    event_tracked = 'CS ON'

    vid_look_up_experiments_cols = {
        "Olena": ["conditioning", "extinction_1", "extinction_2", "retrieval", "late_retrieval", "renewal"],
        "Ozge": ["conditioning", "extinction_1"],
        "Patrick": ["conditioning", "extinction_1", "retrieval"],
    }

    vid_look_up_general_cols = {
        "Olena":["animal_id", "camera_id", "camera_type"],
        "Ozge": ["animal_id"],
        "Patrick": ["animal_id"],
    }

    indv_root_paths = {
        "Olena": os.path.join(ROOT, "Olena_Group"),
        "Ozge": os.path.join(ROOT, "Ozge_Group"),
        "Patrick": os.path.join(ROOT, "Patrick_Group"),
    }

    # go through each person
    for key, val in indv_root_paths.items():
        person = key
        print("**********")
        print(f"{key}")
        print("**********")

        PERSON_ROOT = val
        start_info_filepath = os.path.join(PERSON_ROOT, f"{key.lower()}{start_time_info_suffix}")
        print(f"start_info_filepath: {start_info_filepath}")

        df_start_info = pd.read_csv(start_info_filepath)
        #print(df_start_info.head())
        
        # going through each mouse of this person's dataset
        for experiment in vid_look_up_experiments_cols[key]:
            # find vids using vid_look_up_experiments_cols
            print(experiment)
            for index, row in df_start_info.iterrows():
                print(row)
                look_up_strings = []
                look_up_strings.append(experiment)
                
                # find vids using vid_look_up_experiments_cols
                for col_2 in vid_look_up_general_cols[key]:
                    # don't include when "camera_id" is "0" in the look_up_strings list
                    if col_2 == "camera_id" and row[col_2] == 0:
                        continue # short circuiting the loop
                    look_up_strings.append(row[col_2])

                # making all elements strings
                look_up_strings = [str(item) for item in look_up_strings]
                print(look_up_strings)
                vid_found = find_file_with_strings(PERSON_ROOT, look_up_strings)
                print("vid_found: ", vid_found)
                # based on this level of info, we can get chamber and experimental group info
                # Get fps of video

                vid_opencv_obj = cv2.VideoCapture(vid_found)
                fps = vid_opencv_obj.get(cv2.CAP_PROP_FPS)

                # Print the fps
                print("Frames per second:", fps)
                fps_eztrack_adjusted = fps / 2 # bc eztrack downsamples by 2
                vid_opencv_obj.release()

                chamber = row[chamber_col]
                experimental_group = row[experimental_group_col]
                correction_time_in_secs = row[experiment]
                print("vid info: ", experiment, chamber, experimental_group)
                print("correction_time_in_secs: ", correction_time_in_secs)
                correction_time_in_frames = row[experiment] * fps_eztrack_adjusted
                rounded_correction_time_in_frames = math.ceil(correction_time_in_frames)
                print("correction_time_in_frames: ", correction_time_in_frames)
                
                #to find calibration for extinction_get rid of _1 and _2
                if "_1" in experiment or "_2" in experiment:
                    experiment_cali = experiment[:-2]
                else:
                    experiment_cali = experiment

                print([chamber, experiment_cali])
                calibration_vid_file_name = find_file_with_strings(CALIBRATION_ROOT, [chamber, experiment_cali])
                # need to modify the calibration vids for olena she has two extinctions and like 2 or 3 more types of experiments
                 

                video_dict = {
                    'dpath'   : CALIBRATION_ROOT,  
                    'file'    : calibration_vid_file_name.split("/")[-1],
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
                cap = cv2.VideoCapture(vid_found)
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print("length of vid: ", length)

                vid_d_end = length

                video_dict = {
                    'dpath'   : PERSON_ROOT,  
                    'file'    : vid_found.split("/")[-1],
                    'fpath'   : vid_found,
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

                vid_name_no_ext = vid_found.split("/")[-1].split(".")[0]
                freezing_result_filename = f"{vid_name_no_ext}_FreezingOutput.csv"
                freezing_result_path = vid_found.replace(vid_found.split("/")[-1], freezing_result_filename)

                # this processing includes frame correction from when experiment started
                processed_freezing_out_path = freezing_result_path.replace(".csv", "_processed.csv")

                df_freezing_out = pd.read_csv(freezing_result_path)
                #finding the timing file for this person and experiment
                timing_filepath = find_file_with_strings(PERSON_ROOT, [person.lower(), "FC_info", experiment])
                df_timing = ez.timing_file_processing(find_file_with_strings(PERSON_ROOT, [person.lower(), "FC_info", experiment]), fps_eztrack_adjusted, rounded_correction_time_in_frames)
                df_aligned = ez.freezing_alignment(df_freezing_out, df_timing)

                df_aligned.to_csv(processed_freezing_out_path, index=False)
                    

            # PLOTTING, AFTER DONE PROCESSING ALL VIDS FOR CURR EXPERIMENT
            # Code to continue running goes here
            opsin_group_colors = [mcolors.to_hex((random.random(), random.random(), random.random())), 
                            mcolors.to_hex((random.random(), random.random(), random.random()))]

            df_timing = pd.read_csv(timing_filepath)
            cs_nums = range(1, len(df_timing) + 1)

            fig, ax = plt.subplots()

            experimental_groups_df = df_start_info.loc[:, ["animal_id", "experimental_group"]]
            print(experimental_groups_df.head())
            grouped_df = experimental_groups_df.groupby("experimental_group").agg(lambda x: list(x))
            d_from_df = grouped_df.T.to_dict(orient='list')

            # flatten 2d array
            for key_1, values in d_from_df.items():
                d_from_df[key_1] = values[0]

            # new d will be created of avgs of groups
            d_groups = {}   

            num_mice = 0
            curr_dir = "/".join(vid_found.split("/")[:-1])
            for file in os.listdir(curr_dir):
                if eztrack_output_processed_suffix in file:
                    #mouse_num = int(file.split("_")[0].replace("RRD", ""))
                    # finding the name of animal embedded in the vid name
                    mouse = file.split(".")[0]
                    print(mouse)
                    experimental_group = None

                    # check if mouse is in one of the opsin groups
                    for key_2, values in d_from_df.items():
                        if mouse in d_from_df[key_2]:
                            opsin = key_2
                            
                    print(mouse, ":", experimental_group)
                    
                    df = pd.read_csv(processed_freezing_out_path)
                    frame_lst = list(df["Frame"])

                    timestamps_lst = list(df["Timestamps"])
                    
                    # stamped_lst is the x
                    stamped_lst = ez.overlap_two_lists(frame_lst, timestamps_lst)

                    # modify y to just be binary and not 0 and 100
                    freezing_lst = ez.lst_to_binary_lst(list(df["Freezing"]))
                    #print(freezing_lst)

                    # half_time_window is in seconds
                    x, proportions = ez.bin_data(frame_lst, timestamps_lst,freezing_lst, half_time_window = half_time_window, fps=fps_eztrack_adjusted, event_tracked=event_tracked)
                    #list_of_freezing_props_all_mice.append(proportions)

                    # add to d
                    if experimental_group in d_groups:
                        d_groups[experimental_group].append(proportions)
                    else:
                        d_groups[experimental_group] = []
                        d_groups[experimental_group].append(proportions)

                    num_mice += 1
                
            count = 0
            for key_3 in d_groups:

                # Convert the list of lists to a NumPy array
                array_of_lists = np.array(d_groups[key_3])

                # Calculate the average of the array along the columns (axis=0)
                average = np.mean(array_of_lists, axis=0)
                print(average)

                # Calculate the standard deviation of the array along the columns (axis=0)
                std_deviation = np.std(array_of_lists, axis=0)
                std_error = [std / math.sqrt(num_mice) for std in std_deviation]

                ax.plot(cs_nums, average, label=key_3, color=opsin_group_colors[count])
                plt.errorbar(cs_nums, average, yerr = std_error, fmt='-o', color=opsin_group_colors[count], capsize=3)

                count += 1

            outfilename = f"{experiment}_halftimewdw{half_time_window}_fps{fps_eztrack_adjusted}_plot.png"
            outpath = "/".join(processed_freezing_out_path.split("/")[:-1]) + "/" + outfilename
            print(outpath)
            
            ax.set_title(f"Proportion of Freezing - {experiment} (n={num_mice})")
            ax.set_ylabel(f"Proportion")
            ax.set_xlabel(f"CS #")
            plt.legend()
            fig.savefig(outpath)
            plt.close()
                    

if __name__ == "__main__":
    main()