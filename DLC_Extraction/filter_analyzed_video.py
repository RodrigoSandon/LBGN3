import deeplabcut

CONFIG = r"/home/rory/repos/ppt_dlc/rdt_sessions-PPT-2021-09-08/config.yaml"


def main():
    VIDEO = r"/media/rory/Padlock_DT/Redundant_Backup/BLA-Insc-8/RDT D1/Session-20211021-093007_BLA-INSC-8-RDT-D1/BLA-Insc-8_RDT_D12021-10-21T11_18_16.avi"

    deeplabcut.filterpredictions(CONFIG, VIDEO, filtertype="arima")


main()
