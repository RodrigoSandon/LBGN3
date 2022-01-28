import pandas as pd
import numpy as np


class BehavioralSession:
    def __init__(self, raw_csv_path):
        self.raw_csv_path = raw_csv_path
        self.preprocessed_csv = None

    def preprocess_csv(self):

        """
        Deleting rows under the column "Evnt_Name" that equal 0
        """
        df = pd.read_csv(self.raw_csv_path)
        print("Prev length: ", len(df))
        # print(df.loc[0]["Evnt_Time"], " is of type ", type(df.loc[0]["Evnt_Time"]))
        df = df[df.Evnt_Time != 0]
        print("After filtering: ", len(df))

        """
        Keeping a count of number of trials initiated in the session.
        """

        is_new_trial = (df.Item_Name == "Forced-Choice Trials Begin") | (
            df.Item_Name == "Free-Choice Trials begin"
        )  # series of booleans #old way of defining a trial 11/10/21
        df["is_new_trial"] = is_new_trial  # new column whether it is a new trial or not
        df["is_new_trial"].value_counts()
        df["trial_num"] = np.cumsum(
            df["is_new_trial"]
        )  # counts "True" as 1 and "False" as 0, replacing the cell with the cumulative sum as it iterates through column

        if self.preprocessed_csv == None:
            self.preprocessed_csv = df

    def get_df(self):
        return self.preprocessed_csv
