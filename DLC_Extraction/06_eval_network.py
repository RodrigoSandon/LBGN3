from deeplabcut import evaluate_network

CONFIG_PATH = r"/home/rory/repos/ppt_dlc/rdt_sessions-PPT-2021-09-08/config.yaml"


def main():
    evaluate_network(config=CONFIG_PATH, Shuffles=[1], plotting=True)


if __name__ == "__main__":
    main()

    """
  OPTIONAL PARAMETERS:
  
  Shuffles: list, optional -List of integers specifying the shuffle indices of the training dataset. The default is [1]

  plotting: bool, optional -Plots the predictions on the train and test images. The default is `False`; if provided it must be either `True` or `False`

  show_errors: bool, optional -Display train and test errors. The default is `True`

  comparisonbodyparts: list of bodyparts, Default is all -The average error will be computed for those body parts only (Has to be a subset of the body parts).

  gputouse: int, optional -Natural number indicating the number of your GPU (see number in nvidia-smi). If you do not have a GPU, put None. See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries
    """
