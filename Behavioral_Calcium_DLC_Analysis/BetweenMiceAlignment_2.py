import os
import glob
import pandas as pd
import numpy as np


def find_csv_files(root_path, startswith):

    files = glob.glob(
        os.path.join(root_path, "**", "%s*") % (startswith),
        recursive=True,
    )

    return files


def truncate_csvs_in_root(root_path, name_of_files_to_trunc, len_threshold):

    files_to_truncate = find_csv_files(root_path, name_of_files_to_trunc)
    # finds csv paths to process

    df_database = TableDatabase()

    for file in files_to_truncate:
        table = Table(file, len_threshold)  # make a table object
        # include table into the database (without dropping rows yet)
        table.include_table(drop_row=False)

    print(f"Number of jagged tables before: {df_database._number_of_jagged_dfs}")

    for file in files_to_truncate:
        table = Table(file, len_threshold)
        table.include_table(drop_row=True)
    print(f"Number of jagged tables after: {df_database._number_of_jagged_dfs}")


class TableDatabase(object):
    """Don't want to store all this data in one object, rather, just want to update
    its parameters that are is is meta of tables."""

    _number_of_jagged_dfs = 0


class Table(TableDatabase):
    def __init__(self, df_path, len_threshold):
        # super().__init__()
        self.path = df_path
        self.df = pd.read_csv(df_path)
        self.len_threshold = len_threshold

    def check_if_df_len_equals_thres(self):
        equals_threshold = True
        for col in self.df.columns:
            for count, val in enumerate(list(self.df[col])):
                if str(val) == "nan":
                    # print(f"Column {col} in {self.path} has NaN at row {count}.")
                    equals_threshold = False

        return equals_threshold

    def drop_last_row_df(self):
        self.df = self.df.drop(self.df.tail(1).index)

    def include_table(self, drop_row: bool):
        equals_threshold = self.check_if_df_len_equals_thres()
        if drop_row == True:  # indicates whether we want to acc drop rows yet
            if (
                equals_threshold is True
            ):  # if the df is not jagged, still check if it needs to be truncated

                self.truncate_past_len_threshold()
            elif equals_threshold is False:  # if it is jagged, do everything
                print(f"File {self.path} is jagged.")
                self.drop_last_row_df()
                TableDatabase._number_of_jagged_dfs -= 1
                self.save_table()

        elif drop_row == False:
            if equals_threshold is True:  # don't do anything if the df is not jagged
                pass
            elif (
                equals_threshold is False
            ):  # if it is jagged, do everything except dropping
                TableDatabase._number_of_jagged_dfs += 1

    def save_table(self):
        new_path = self.path.replace(".csv", "_truncated.csv")
        self.df.to_csv(new_path, index=False)

    def truncate_past_len_threshold(self):

        if len(self.df) > self.len_threshold:
            print(
                f"File {self.path} is not jagged, but higher than threshold: {len(self.df)}"
            )
            # minus 1 bc we refer to index
            self.df = self.df.truncate(after=self.len_threshold - 1)
            print(f"LENGTH AFTER: {len(self.df)}")
            self.save_table()
        elif len(self.df) < self.len_threshold:
            print(f"File {self.path} less than threshold: {len(self.df)}")


def main():
    ROOT_PATH = r"/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData"
    # ROOT_PATH = r"/Users/rodrigosandon/Documents/GitHub/LBGN/SampleData/truncating_bug"

    truncate_csvs_in_root(
        ROOT_PATH, name_of_files_to_trunc="all_concat_cells.csv", len_threshold=200
    )


main()
