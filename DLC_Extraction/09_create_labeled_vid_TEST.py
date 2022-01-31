from deeplabcut import create_labeled_video
from pathlib import Path
from typing import List

from numpy import save

CONFIG_PATH = r"/home/rory/repos/ppt_dlc2/config.yaml"


def find_video_paths(
    raw_data_parent_dir: Path, dir_of_vids_for_training: Path
) -> List[str]:

    videos_of_training = [
        str(vid) for i, vid in enumerate(dir_of_vids_for_training.glob("*.avi"))
    ]
    return videos_of_training
    return [
        str(vid)
        for i, vid in enumerate(raw_data_parent_dir.glob("*.avi"))
        if str(vid) not in videos_of_training
    ]


RAW_DATA_DIR = Path(r"/home/rory/repos/ppt_dlc/rdt_sessions-PPT-2021-09-08/videos")
VIDS_OF_TRAINING = Path(r"/home/rory/repos/ppt_dlc/rdt_sessions-PPT-2021-09-08/videos")
video_paths = find_video_paths(RAW_DATA_DIR, dir_of_vids_for_training=VIDS_OF_TRAINING)


def main():
    # print("Creating labeled video for: ", video_paths[5])
    # create_labeled_video(CONFIG_PATH, video_paths[5], save_frames=True)
    create_labeled_video(
        CONFIG_PATH,
        [
            "/media/rory/Padlock_DT/Novel_Video_Analysis_2/BLA-Insc-9_RDT_D12021-10-26T11_46_45.avi"
        ],
        filtered=False,
        draw_skeleton=True,
        save_frames=False,
    )


if __name__ == "__main__":
    main()
