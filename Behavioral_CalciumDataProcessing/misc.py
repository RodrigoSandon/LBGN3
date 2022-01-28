import os
from pathlib import Path


def get_session_name(path):
    name = path.split("/")[-1].replace(".csv", "")
    return name


ROOT_MOUSE = Path(r"/media/rory/PTP Inscopix 2/PTP_Inscopix_#3/BLA-Insc-6")
"""Give a list of the strings that have another string --> these ones don't include in analysis"""
to_not_include_in_preprocessing = [
    "_ABET_processed.csv",
    "_ABET_GPIO_processed.csv",
    "resnet50",
]

for root, dirs, files in os.walk(ROOT_MOUSE, topdown=True):
    roots_not_included = []
    if root.find("Session") != -1:
        # only roots with "Session"
        if (
            root.find("SingleCellAlignmentData") != -1
            or root.find("BetweenCellAlignmentData") != -1
        ):  # found
            roots_not_included.append(root)
        else:
            for name in files:
                if name.startswith("BLA"):

                    if any(
                        os.path.join(root, name).find(session) != -1
                        for session in to_not_include_in_preprocessing
                    ):  # for any of the list elements that are not found in the current name of file
                        pass
                    else:
                        curr_session = os.path.join(root, name)
                        print(curr_session)
                        ses_name = get_session_name(curr_session)
                        print(ses_name)

    # print("roots not included:", roots_not_included)
"""for root, dirs, files in os.walk(ROOT_MOUSE, topdown=True):
    if root.find("Session") != -1:
        for name in files:
            if name.startswith("BLA"):
                print(os.path.join(root, name))
"""
