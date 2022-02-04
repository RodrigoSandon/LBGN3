# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 15:27:07 2022

@author: vlcor
"""
import os
import glob
import h5py
import numpy as np
from scipy.signal import savgol_filter
import pickle

#Takes in an h5 predicted label file of an entire video

vidfolder = r"T:\SLEAP\OdorPilots\Round2\cropped" #this should point to the folder where all your slp/h5 files are
h5path = os.path.join(vidfolder, '*.h5')

files = sorted(glob.glob(h5path)) #isolate only the h5 files in thise folder

print("number of files:", len(files))

# conversion factors - I use this data to convert my SLEAP position data into cm and s (instead of pixels and frames)
xrange_pix = [74,733]; #pix in video - obv you'll want to change this based on your videos and chamber
xrange_cm = [0,36.5]; #cm
posconvfactor = xrange_cm[1]/(xrange_pix[1] - xrange_pix[0]); #cm/pix
timeconvfactor = 100; #frames/s

#this loop will cycle through all of your h5 files and extract the joint data
jointsdict = dict() ##will save here
for f in range(len(files)):
    
    #get filename
    filename = files[f]
    
    #generate sessionname from full filename
    #these idx details will obviously differ depending on your filenames
    startidx = len(vidfolder)+1
    stopidx = 18
    sessionname = filename[startidx:-stopidx]
    
    #open h5 file and pull out the joints "tracks"
    currfile = h5py.File(filename, 'r')
    joints = np.array(currfile['tracks'])
    
    #for some reason the joints data has an extra 4th dimension, so I get rid of that here
    jointsreduce=np.squeeze(joints)
    #axis0 = x/y coordinates
    #axis1 = body part list (this will go in order according to how you set up your SLEAP file)
    #axis2 = time
    
    #smooth joints data - 2nd and 3rd parameters here are simply chosen based on my frame rate and visual inspection of the smoothing
    #might need to adjust details for your data
    jointsmooth = savgol_filter(jointsreduce, 49, 3, axis=2)
    
    #I typically try to save the tracking data (and things like velocity, center position, etc) in dictionaries where the keys are the sessionnames
    #going through all the files and extracting the h5 can take some time with a lot of files, so it's faster to just extract once and save what you need for later
    jointsdict[sessionname] = jointsmooth


##save all extracted data (details will vary here, depending on what you want to save)
savename = 'OdorPilotsTake2Data.pkl'
f = open(savename,"wb")
pickle.dump([jointsdict, otherdict, otherdict, etc],f)
f.close()

