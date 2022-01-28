from deeplabcut import create_labeled_video
from pathlib import Path
from typing import List

from numpy import save

CONFIG_PATH = r"/home/rory/repos/ppt_dlc/rdt_sessions-PPT-2021-09-08/config.yaml"


def main():
    print(
        "Creating labeled video for: ",
        "/media/rory/RDT VIDS/DeepLabCut_RDT_Sessions_Only/100_RDT_D12020-01-16T12_26_05.avi",
    )

    create_labeled_video(
        CONFIG_PATH,
        [
            "/media/rory/RDT VIDS/DeepLabCut_RDT_Sessions_Only/100_RDT_D12020-01-16T12_26_05.avi"
        ],
        save_frames=True,
        draw_skeleton=True,
    )


if __name__ == "__main__":
    main()
