import pandas as pd
import numpy as np
import os, glob
from pathlib import Path
from ShockUtilities import ShockUtilities
from typing import List

class ShockSession:
    def __init__(self, name, raw_csv_path):
        self.name = name
        self.raw_csv_path = raw_csv_path
        self.preprocessed_csv = None

    def preprocess_csv(self):

        df = pd.read_csv(self.raw_csv_path)
        print("Prev length: ", len(df))

        df = df[df.Evnt_Time != 0]
        print("After filtering for housekeeping trial: ", len(df))
        # print(df.head())

        is_new_trial = df.Item_Name == "Pulse Shock"
        df["is_new_trial"] = is_new_trial  # new column whether it is a new trial or not
        df["is_new_trial"].value_counts()
        print(f"Number of trials in shock session:")
        print(df["is_new_trial"].value_counts())

        df["trial_num"] = np.cumsum(
            df["is_new_trial"]
        )  # counts "True" as 1 and "False" as 0, replacing the cell with the cumulative sum as it iterates through column

        if self.preprocessed_csv == None:
            self.preprocessed_csv = df

    def get_df(self):
        return self.preprocessed_csv


def find_paths_startswith_and_endswith(root_path, startswith, endswith) -> List:

    files = glob.glob(
        os.path.join(root_path, "**", "%s*%s") % (startswith, endswith),
        recursive=True,
    )

    return files


def main():

    ROOT = r"/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5"

    to_not_include_in_preprocessing = [
        "_ABET_processed.csv",
        "_ABET_GPIO_processed.csv",
        "resnet50",
    ]  # can't include these in the file name

    files = find_paths_startswith_and_endswith(ROOT, "BLA", ".csv")
    acc_files = []
    for f in files:
        df = pd.read_csv(f)
        if list(df.columns)[0] == "Evnt_Time":
            acc_files.append(f)
    print(len(files))
    print(len(acc_files))

    for abet_path in acc_files:

        if any(
            abet_path.find(mystr) != -1 for mystr in to_not_include_in_preprocessing
        ):  # for any of the list elements to not include that are found in the current name of file, pass
            pass
        else:
            if abet_path.find("Shock Test") != -1:

                print("CURRENT PATH: ", abet_path)
                ses_name = "Shock Test"

                ABET_1 = ShockSession(
                    ses_name,
                    abet_path,
                )
                ABET_1.preprocess_csv()
                df = ABET_1.get_df()
                grouped_by_trialnum = df.groupby("trial_num")
                processed_behavioral_df = grouped_by_trialnum.apply(
                    ShockUtilities.process_csv
                )  # is a new df, it's not the modified df

                processed_behavioral_df = ShockUtilities.del_first_row(
                    processed_behavioral_df
                )

                processed_behavioral_df = ShockUtilities.add_shock_intensity(
                    processed_behavioral_df
                )

                new_path = abet_path.replace(".csv", "_ABET_processed.csv")
                processed_behavioral_df.to_csv(
                    new_path,
                    index=True,
                )


if __name__ == "__main__":
    main()
