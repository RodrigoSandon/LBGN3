import os
import holoviews as hv
import numpy as np
import pandas as pd
import EzTrack.FreezeAnalysis_Functions as fz


dpath = r"/media/rory/Padlock_DT/Fear_Conditioning_Control/NewVideos/Conditioning/"
vid_name = "RRD276_C18457_Conditioning.avi"
chamber = None
mt_cutoff = 8.73573982600555

h,w = 300,1000 

FreezeThresh = 180 
MinDuration = 40 

vid_d_start = 0
vid_d_end = 32396
dsmpl = 1

disp_d_start = 1
disp_d_end = 10000
fps = 30


video_dict = {
    'dpath'   : dpath,  
    'file'    : vid_name,
    'start'   : vid_d_start, 
    'end'     : vid_d_end,
    'dsmpl'   : dsmpl,
    'stretch' : dict(width=1, height=1)
}

img_crp, video_dict = fz.LoadAndCrop(video_dict, cropmethod="Box")
img_crp

Motion = fz.Measure_Motion(video_dict, mt_cutoff, SIGMA=1)  
print(len(Motion))
plt_mt = hv.Curve((np.arange(len(Motion)),Motion),'Frame','Pixel Change').opts(
    height=h,width=w,line_width=1,color="steelblue",title="Motion Across Session")
plt_mt

Freezing = fz.Measure_Freezing(Motion,FreezeThresh,MinDuration)  
fz.SaveData(video_dict,Motion,Freezing,mt_cutoff,FreezeThresh,MinDuration)
print('Average Freezing: {x}%'.format(x=np.average(Freezing)))

plt_fz = hv.Area(Freezing*(Motion.max()/100),'Frame','Motion').opts(
    color='lightgray',line_width=0,line_alpha=0)
plt_mt = hv.Curve((np.arange(len(Motion)),Motion),'Frame','Motion').opts(
    height=h,width=w,line_width=1, color='steelblue',
    title="Motion Across Session with Freezing Highlighted in Gray")
plt_fz*plt_mt*hv.HLine(y=FreezeThresh).opts(color='red',line_width=1,line_dash='dashed')

display_dict = {
    'start'      : disp_d_start, 
    'end'        : disp_d_end,
    'fps'        : fps,
    'resize'     : None,
    'save_video' : True
}

fz.PlayVideo(video_dict,display_dict,Freezing,mt_cutoff,SIGMA=1)