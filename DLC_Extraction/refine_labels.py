import deeplabcut as dlc


def main():
    CONFIG = (
        r"/home/rory/Rodrigo/DLC_Extraction/rdt_sessions-PPT-2021-09-08/config.yaml"
    )

    dlc.refine_labels(CONFIG)


if __name__ == "__main__":
    main()
