from deeplabcut import analyze_videos
from pathlib import Path
from typing import List

CONFIG_PATH = r"/home/rory/repos/ppt_dlc2/config.yaml"


def main():

    analyze_videos(
        CONFIG_PATH,
        [
            "/media/rory/Padlock_DT/DeepLabCut_RDT_Sessions_Only/BLA-Insc-9_RDT_D12021-10-26T11_46_45.avi"
        ],
        save_as_csv=True,
        destfolder="/media/rory/Padlock_DT/Novel_Video_Analysis_2",
    )


if __name__ == "__main__":

    main()
