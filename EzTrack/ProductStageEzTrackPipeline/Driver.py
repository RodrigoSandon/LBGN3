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

def main():

    FreezeThresh = 180 
    MinDuration = 40

    experiment_types = ["Extinction", "Retrieval", "Conditioning"]

    for experiment_type in experiment_types:
    
        correspondece_file = "mouse_chamber_corrrespondence.csv"
        colname_vid_paths = "mouse_vid_path"

        root_calibration_vids = f"/media/rory/Padlock_DT/Fear_Conditioning_Control/NewVideos/Calibration"

        ROOT = f"/media/rory/Padlock_DT/Fear_Conditioning_Control/NewVideos/{experiment_type}"

        correspondence_filepath = os.path.join(ROOT, correspondece_file)

        df_correspondence = pd.read_csv(correspondence_filepath)

        vid_list = df_correspondence[colname_vid_paths]

        for vid in vid_list:
            print(f"vid: {vid}")

            vid_name = vid.split("/")[-1]

            chamber = df_correspondence.query(f"{colname_vid_paths}=='{vid}'")["chamber"]
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
                'start'   : 0, 
                'end'     : None,
                'dsmpl'   : 1,
                'stretch' : dict(width=1, height=1),
                'cal_frms' : 600
                }


            img_crp, video_dict = fz.LoadAndCrop(video_dict)

            ####### CALIBRATION #######
            cal_dif_avg, percentile, mt_cutoff = fz.calibrate_custom(video_dict, cal_pix=10000, SIGMA=1)

            ####### FREEZE ANALYSIS #######
            cap = cv2.VideoCapture(vid)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("length of vid: ", length)

            h,w = 300,1000  

            vid_d_start = 0
            vid_d_end = length
            dsmpl = 1

            disp_d_start = 1
            disp_d_end = 600
            fps = 30


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
    
    
    experiment_types = ["Extinction", "Retrieval","Conditioning"]
    
    for experiment_type in experiment_types:

        ROOT_TIMING_FILE = "/media/rory/Padlock_DT/Fear_Conditioning_Control/"
        timing_file_name = f"{experiment_type}_CS_timing_FC_Control.csv"
        timing_filepath = os.path.join(ROOT_TIMING_FILE, timing_file_name)
        ROOT = f"/media/rory/Padlock_DT/Fear_Conditioning_Control/NewVideos/{experiment_type}"

        fps = 30

        for file in os.listdir(ROOT):
            if "FreezingOutput.csv" in file:
                file_path = os.path.join(ROOT, file)
                file_out_path = file_path.replace(".csv", "_processed.csv")
                print(file_path)

                df_freezing_out = pd.read_csv(file_path)
                #df_freezing_out = freezing_output_processing(file_path)
                df_timing = timing_file_processing(timing_filepath, fps)
                df_aligned = freezing_alignment(df_freezing_out, df_timing)

                df_aligned.to_csv(file_out_path, index=False)




if __name__ == "__main__":
    main()