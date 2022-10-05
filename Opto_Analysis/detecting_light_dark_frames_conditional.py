import cv2
import imageio
import numpy as np
import os, glob
from csv import writer, DictWriter


"""
instead of having a set pre-defined (hardcoded) threshold, if mean brightness is 3 factors greater than the previous one,
the light has been turned on (this is still hard-coding it but a little less hard coded)
"""
def img_estim(mean_brightness_curr, mean_brightness_prev):
    #print(f"mean: {mean_brightness_curr}")
    is_light = mean_brightness_curr >= mean_brightness_prev*3
    return "light" if is_light else "dark"

def dark_to_light(prev_frame, curr_frame):
    return "switch" if prev_frame == "dark" and curr_frame == "light" else "no_switch" 

def file_name_parsing(filename: str):
    filename = filename.split("/")[-1]
    description = filename.replace(
        "_merged_resized_grayscaled.mp4", "")
    return description

def find_paths_endswith(root_path, endswith) -> list:

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )

    return files

#vid_path = "/media/rory/RDT VIDS/BORIS/RRD171_RDT_OPTO_OUTCOME_01112021_5_merged_resized_grayscaled.mp4"
#vid_path = "/media/rory/RDT VIDS/BORIS/RRD22_Risk_0.1_mA_OPTO_ALL_FREE_OUTCOMES_01102019_2_merged_resized_grayscaled.mp4"

result_folder = "/media/rory/Padlock_DT/Opto_Speed_Analysis/detecting_light_dark_frames"

ROOT_1 = r"/media/rory/RDT VIDS/BORIS/"
ROOT_2 = r"/media/rory/RDT VIDS/BORIS_merge/"
ROOT_3 = r"/media/rory/RDT VIDS/BORIS_merge/BATCH_2"

vid_paths_1 = find_paths_endswith(ROOT_1, "merged_resized_grayscaled.mp4")
vid_paths_2 = find_paths_endswith(ROOT_2, "merged_resized_grayscaled.mp4")
vid_paths_3 = find_paths_endswith(ROOT_3, "merged_resized_grayscaled.mp4")

vid_paths = vid_paths_1 + vid_paths_2 + vid_paths_3

print(vid_paths)
csv_name = "/media/rory/Padlock_DT/Opto_Speed_Analysis/detecting_light_dark_frames/opto_abet_file_corrections.csv"
header = ["vid_path", "ABET_addition_correction_time_(s)"]

for vid_path in vid_paths:
    print(vid_path)
    vidcap = cv2.VideoCapture(vid_path)

    success, image = vidcap.read()
    #dark_to_light_thrshld = 8
    count = 0

    prev_frame_est = None
    prev_frame_mean = None
    curr_frame_est = None
    # while a frame read is successful and while the time for a the experiment to start is below 2 mins (120secs*30frames)
    while success and count < 3600:
        descrp = file_name_parsing(vid_path)
        frame_image_path = os.path.join(result_folder, f"{descrp}_frame_{count}_{count/30}secs.jpg")
        #f = imageio.imread(frame_image_path, as_gray=True)

        curr_frame_mean = np.mean(image)

        if count > 0:
            curr_frame_est = img_estim(curr_frame_mean, prev_frame_mean)
            result = dark_to_light(prev_frame_est, curr_frame_est)
            if result == "switch":  
                cv2.imwrite(frame_image_path, image)
                print(f"{result} found at frame {count}, time {count/30} secs. Comparison was {prev_frame_mean} -> {curr_frame_mean}")

                data = [str(vid_path), count/30]
                # look if the csv for this trial exists already
                if os.path.exists(csv_name) == True:
                    with open(csv_name, "a") as csv_obj:
                        writer_obj = writer(csv_obj)
                        writer_obj.writerow(data)
                        csv_obj.close()

                # else (if the csv doesn't exist):
                # make new csv, add the header row (cell + timepoints in a list), and append data
                else:
                    with open(csv_name, "w+") as csv_obj:
                        writer_obj = writer(csv_obj)
                        writer_obj.writerow(header)
                        writer_obj.writerow(data)
                        csv_obj.close()

                break
    
        success, image = vidcap.read()
        count += 1
        print(count)
        #print(f"frame: {count} | {img_estim(curr_frame_mean, 21)}")
        prev_frame_est = curr_frame_est
        prev_frame_mean = curr_frame_mean