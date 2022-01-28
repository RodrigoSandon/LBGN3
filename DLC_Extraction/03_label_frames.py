from deeplabcut import label_frames

CONFIG_PATH = r"/home/rory/repos/ppt_dlc/rdt_sessions-PPT-2021-09-08/config.yaml"


def main():
    label_frames(config=CONFIG_PATH)


if __name__ == "__main__":
    main()
