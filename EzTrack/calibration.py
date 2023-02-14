import os
import holoviews as hv
import numpy as np
import pandas as pd
import FreezeAnalysis_Functions as fz

def parse_cali_vid(mystr: str):
    chamber = mystr.split("_")[1]
    return chamber

def main():
    cal_dif_avg: float # Average frame-by-frame pixel difference
    percentile: float # 99.99 percentile of pixel change differences
    mt_cutoff: float # Grayscale change cut-off for pixel change

    # root folder where your calibration videos are found
    ROOT = r"/media/rory/Padlock_DT/Fear Conditioning Control/NewVideos/Calibration"

    for cali_vid in os.listdir(ROOT):
        if ".avi" in cali_vid:
            # assumes the chamber is denoted in it's name
            chamber = parse_cali_vid(cali_vid)

            video_dict = {
            'dpath'   : ROOT,  
            'file'    : cali_vid,
            'start'   : 0, 
            'end'     : None,
            'dsmpl'   : 1,
            'stretch' : dict(width=1, height=1),
            'cal_frms' : 600
            }


            img_crp, video_dict = fz.LoadAndCrop(video_dict)

            cal_dif_avg, percentile, mt_cutoff = fz.calibrate_custom(video_dict, cal_pix=10000, SIGMA=1)
            

if __name__ == "__main__":
    main()