import pandas as pd
import numpy as np


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
