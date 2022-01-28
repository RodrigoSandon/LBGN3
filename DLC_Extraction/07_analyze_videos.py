from deeplabcut import analyze_videos
from pathlib import Path
from typing import List
import os, glob

CONFIG_PATH = r"/home/rory/repos/ppt_dlc/rdt_sessions-PPT-2021-09-08/config.yaml"

# Test version: only analyzing one video


def find_video_paths(raw_data_parent_dir: Path) -> List[str]:

    videos = [str(vid) for i, vid in enumerate(raw_data_parent_dir.glob("*.avi"))]
    print(*videos, sep="\n")
    print(len(videos))
    return videos
    # return [
    #   str(vid)
    #    for i, vid in enumerate(raw_data_parent_dir.glob("*.avi"))
    #    if str(vid) not in videos_of_training
    # ]


def find_paths_endswith(root_path, endswith) -> list:

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )

    return files


RAW_DATA_DIR = Path(r"/media/rory/Padlock_DT/Redundant_Backup")
video_paths = find_paths_endswith(RAW_DATA_DIR, ".avi")

# print(video_paths)
# excluding videos that were used for training in our analysis
def main():
    for i in video_paths:
        print("ANALYZING: ", i)
        analyze_videos(
            CONFIG_PATH,
            i,
            destfolder="/media/rory/Padlock_DT/DeepLabCut_RDT_Sessions_Only",
            save_as_csv=True,
        )


if __name__ == "__main__":

    main()
