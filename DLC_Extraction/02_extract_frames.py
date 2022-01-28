from deeplabcut import extract_frames
from pathlib import Path

CONFIG_PATH = r"/home/rory/repos/ppt_dlc/rdt_sessions-PPT-2021-09-08/config.yaml"


def main():
    extract_frames(config=CONFIG_PATH, mode="automatic", userfeedback=False)


if __name__ == "__main__":
    main()
