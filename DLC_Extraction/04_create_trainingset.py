from deeplabcut import create_training_dataset

CONFIG_PATH = r"/home/rory/repos/ppt_dlc2/config.yaml"


def main():
    create_training_dataset(config=CONFIG_PATH, augmenter_type="imgaug")


if __name__ == "__main__":
    main()
