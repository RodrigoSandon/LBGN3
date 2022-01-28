import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from typing import List


def find_paths_startswith_and_endswith(root_path, startswith, endswith) -> List:

    files = glob.glob(
        os.path.join(root_path, "**", "%s*%s") % (startswith, endswith),
        recursive=True,
    )

    return files


def main():

    ROOT = r"/media/rory/Padlock_DT/BLA_Analysis"

    files = find_paths_startswith_and_endswith(
        ROOT, "BLA", "_ABET_GPIO_processed.csv")

    for abet_path in files:

        if abet_path.find("Shock Test") != -1:  # found

            print("CURRENT PATH: ", abet_path)
            df = pd.read_csv(abet_path)

            new_col_list = []
            for num in range(0, 6):
                new_col_list.append("0-0.1")
            for num in range(0, 5):
                new_col_list.append("0.12-0.2")
            for num in range(0, 5):
                new_col_list.append("0.22-0.3")
            for num in range(0, 5):
                new_col_list.append("0.32-0.4")
            for num in range(0, 5):
                new_col_list.append("0.42-0.5")

            df["Bin"] = new_col_list
            new_name = abet_path.replace(
                "_ABET_GPIO_processed.csv", "_ABET_GPIO_groupby_processed.csv")
            df.to_csv(new_name, index=False)


if __name__ == "__main__":
    main()
