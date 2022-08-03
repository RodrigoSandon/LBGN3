import cv2
import imageio
import numpy as np
import os, glob

def img_estim(mean_brightness, thrshld):
    #print(f"mean: {mean_brightness}")
    is_light = mean_brightness > thrshld
    return "light" if is_light else "dark"

def dark_to_light(prev_frame, curr_frame):
    return "switch" if prev_frame == "dark" and curr_frame == "light" else "no_switch" 

result_folder = "/media/rory/Padlock_DT/Opto_Analysis/detecting_light_dark_frames"
#vid_path = "/media/rory/RDT VIDS/BORIS/RRD171_RDT_OPTO_OUTCOME_01112021_5_merged_resized_grayscaled.mp4"
vid_path = "/media/rory/RDT VIDS/BORIS/RRD22_Risk_0.1_mA_OPTO_ALL_FREE_OUTCOMES_01102019_2_merged_resized_grayscaled.mp4"

vidcap = cv2.VideoCapture(vid_path)

success, image = vidcap.read()
dark_to_light_thrshld = 8
count = 0

prev_frame_est = None
prev_frame_mean = None
while success:
    frame_image_path = os.path.join(result_folder, f"frame_{count}.jpg")
    cv2.imwrite(frame_image_path, image)
    f = imageio.imread(frame_image_path, as_gray=True)

    curr_frame_mean = np.mean(f)
    curr_frame_est = img_estim(curr_frame_mean, 21)
    if count > 0:
        result = dark_to_light(prev_frame_est, curr_frame_est)
        if result == "switch":
            print(f"{result} found at frame {count}, time {count/30} secs. Comparison was {prev_frame_mean} -> {curr_frame_mean}")
            break
   
    success, image = vidcap.read()
    count += 1
    print(count)
    #print(f"frame: {count} | {img_estim(curr_frame_mean, 21)}")
    prev_frame_est = curr_frame_est
    prev_frame_mean = curr_frame_mean