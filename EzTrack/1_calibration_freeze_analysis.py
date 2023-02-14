import cv2
import os, glob
import pandas as pd
import numpy as np
import holoviews as hv
import FreezeAnalysis_Functions as fz

def main():

    FreezeThresh = 180 
    MinDuration = 40

    experiment_type = "Extinction"
    correspondece_file = "mouse_chamber_corrrespondence.csv"
    colname_vid_paths = "mouse_vid_path"

    root_calibration_vids = f"/media/rory/Padlock_DT/Fear_Conditioning_Control/NewVideos/Calibration"

    ROOT = f"/media/rory/Padlock_DT/Fear_Conditioning_Control/NewVideos/{experiment_type}"

    correspondence_filepath = os.path.join(ROOT, correspondece_file)

    df_correspondence = pd.read_csv(correspondence_filepath)

    vid_list = df_correspondence[colname_vid_paths]

    for vid in vid_list:

        vid_name = vid.split("/")[-1]

        chamber = df_correspondence.query(f"{colname_vid_paths}=='{vid}'")["chamber"]
        chamber = chamber.values[0]

        # the calibration video must contain the two: experiment type and chamber
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
            'fpath'   : vid,
            'dpath'   : ROOT,  
            'file'    : vid_name,
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

        """plt_fz = hv.Area(Freezing*(Motion.max()/100),'Frame','Motion').opts(
            color='lightgray',line_width=0,line_alpha=0)
        plt_mt = hv.Curve((np.arange(len(Motion)),Motion),'Frame','Motion').opts(
            height=h,width=w,line_width=1, color='steelblue',
            title="Motion Across Session with Freezing Highlighted in Gray")
        plt_fz*plt_mt*hv.HLine(y=FreezeThresh).opts(color='red',line_width=1,line_dash='dashed')"""

        """display_dict = {
            'start'      : disp_d_start, 
            'end'        : disp_d_end,
            'fps'        : fps,
            'resize'     : None,
            'save_video' : True
        }

        fz.PlayVideo(video_dict,display_dict,Freezing,mt_cutoff,SIGMA=1)
        """
        vid_name_no_ext = vid_name.split(".")[0]
        result_filename = f"{vid_name_no_ext}_FreezingOutput.csv"
        result_path = os.path.join(ROOT, result_filename)





        break

if __name__ == "__main__":
    main()