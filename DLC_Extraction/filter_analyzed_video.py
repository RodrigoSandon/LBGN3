import deeplabcut

CONFIG = r"/home/rory/repos/ppt_dlc2/config.yaml"


def main():
    VIDEO = r"/media/rory/Padlock_DT/Novel_Video_Analysis_2/BLA-Insc-9_RDT_D12021-10-26T11_46_45.avi"

    deeplabcut.filterpredictions(CONFIG, VIDEO, filtertype="arima")


main()
