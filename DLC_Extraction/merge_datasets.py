import deeplabcut as dlc


def main():

    CONFIG = r"/home/rory/repos/ppt_dlc/rdt_sessions-PPT-2021-09-08/config.yaml"

    dlc.merge_datasets(CONFIG)


main()
