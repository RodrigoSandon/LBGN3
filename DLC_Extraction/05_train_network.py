from deeplabcut import train_network

CONFIG_PATH = r"/home/rory/repos/ppt_dlc2/config.yaml"


def main():
    train_network(config=CONFIG_PATH, maxiters=1000000)


if __name__ == "__main__":
    main()
