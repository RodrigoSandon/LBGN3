from deeplabcut import analyze_videos
from pathlib import Path
from typing import List

CONFIG_PATH = r"/home/rory/repos/ppt_dlc/rdt_sessions-PPT-2021-09-08/config.yaml"


def main():

    analyze_videos(
        CONFIG_PATH,
        [
            "/media/rory/Padlock_DT/Redundant_Backup/BLA-Insc-9/RDT D1/Session-20211026-102032_bla_insc_9_rdt_d1/BLA-Insc-9_RDT_D12021-10-26T11_46_45.avi"
        ],
        save_as_csv=True,
    )


if __name__ == "__main__":

    main()
