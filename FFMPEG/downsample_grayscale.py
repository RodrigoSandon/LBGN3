import os
import glob
from pathlib import Path

def find_paths_endswith(root_path, endswith) -> list:

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )
    return files

def resize_video(video_path, new_w, new_h, out_path):
    #ffmpeg -i input.mp4 -vf scale=$w:$h <encoding-parameters> output.mp4
    cmd = f"ffmpeg -i {video_path} -vf scale={new_w}:{new_h} -preset slow -crf 18 {out_path}"
    os.system(cmd)

def grayscale_video(video_path, out_path):
    cmd = f"ffmpeg -i {video_path} -vf hue=s=0 {out_path}"
    os.system(cmd)

def main():
    """
    Comparisons:
    1) Original, unedited
    2) Reduced dimensions
    3) Reduced dimensions + grayscaled
    4) Reduced dimensions + grayscaled + reduced temporal resolution

    Thereafter:
     - Predict (processed) files and see how sleap does in predicting them via the GUI
    """
    ROOT_PATH = r"/media/rory/RDT VIDS/BORIS"
    video_paths = find_paths_endswith(ROOT_PATH, "merged.mp4")

    for video_path in video_paths:
        resize_out_path = video_path.replace(".mp4", "_resized.mp4")
        grayscale_out_path = resize_out_path.replace(".mp4", "_grayscaled.mp4")
        if os.path.isfile(resize_out_path) == False and os.path.isfile(grayscale_out_path) == False:

            video_path = f"'{video_path}'" ## fmpeg doesn't like whitespace, so quote it
            print(video_path)
            resize_video(video_path, 800, 600, resize_out_path)
            grayscale_video(resize_out_path, grayscale_out_path)

def one_vid():

    video_path = "/media/rory/Padlock_DT/Opto_Analysis/RRD19_Risk_0.1_mA_OPTO_ALL_FREE_OUTCOMES_01092019_2_merged.mp4"

    resize_out_path = video_path.replace(".mp4", "_resized.mp4")
    resize_video(video_path, 800, 600, resize_out_path)

    grayscale_out_path = resize_out_path.replace(".mp4", "_grayscaled.mp4")
    grayscale_video(resize_out_path, grayscale_out_path)
    

if __name__ == "__main__":
    main()