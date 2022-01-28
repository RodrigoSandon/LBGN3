from deeplabcut import add_new_videos

CONFIG_PATH = r"/home/rory/repos/ppt_dlc/rdt_sessions-PPT-2021-09-08/config.yaml"


def main():
    videos_list = [
        "/media/rory/Padlock_DT/Redundant_Backup/BLA-Insc-11/PR D1/Session-20211108-113157/BLA-INSC-11_PR_D12021-11-08T12_56_36.avi",
    ]
    add_new_videos(CONFIG_PATH, videos_list)


if __name__ == "__main__":
    main()
