import deeplabcut as dlc

# you can ind the labeled frames in /home/rory/repos/ppt_dlc2
def main():
    CONFIG = r"/home/rory/repos/ppt_dlc2/config.yaml"

    dlc.refine_labels(CONFIG)


if __name__ == "__main__":
    main()
