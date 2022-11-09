import numpy as np
import glob
import os
from typing import List
import pandas as pd
from pathlib import Path

def find_paths(root_path: Path, middle: str, endswith: str) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, "**", middle, "**", endswith), recursive=True,
    )
    return files

#/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis/BLA_NAcShell/ArchT/Choice/RRD16/body/AlignmentData/Block_Trial_Type_Reward_Size_Start_Time_(s)/(1.0, 'Forced', 'Large')/speeds_z_-5_5savgol_avg.csv
def main():


    session_root = r"/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis"

    list_of_combos_we_care_about = [

            "Block_Trial_Type_Start_Time_(s)",
        ]

    filename = "speeds.csv"

    for combo in list_of_combos_we_care_about:

        files = find_paths(session_root, combo ,filename)


        for csv in files:
            mouse = csv.split("/")[9]

            df: pd.DataFrame
            df = pd.read_csv(csv)

            # transpose df
            df = df.T
            # remove first row
            df = df.iloc[1:, :]
            # change col names: mouse_trail#
            num_cols = len(list(df.columns))
            for i in range(0, num_cols):
                df = df.rename(columns={i : f"{mouse}_Trail_{i + 1}"})

            # set col name for index
            df.index.name = "Time_(s)"

            new_name = csv.replace(".csv", "_renamed.csv")
            df.to_csv(new_name)

def one_process():


    csv = r"/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis/BLA_NAcShell/ArchT/choice/RRD16/body/AlignmentData/Block_Trial_Type_Start_Time_(s)/(1.0, 'Free')/speeds_z_-5_5savgol.csv"

    mouse = csv.split("/")[9]

    df: pd.DataFrame
    df = pd.read_csv(csv)

    # transpose df
    df = df.T
    print(df.head())
    # remove first row
    df = df.iloc[1:, :]
    # change col names: mouse_trail#
    num_cols = len(list(df.columns))
    for i in range(0, num_cols):
        df = df.rename(columns={i : f"{mouse}_Trail_{i + 1}"})

    # set col name for index
    df.index.name = "Time_(s)"

    print(df.head())

    new_name = csv.replace(".csv", "_renamed.csv")
    df.to_csv(new_name)


if __name__ == "__main__":
    main()
    #one_process()