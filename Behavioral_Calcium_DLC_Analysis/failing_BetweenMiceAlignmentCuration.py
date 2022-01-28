import os, glob
import pandas as pd
import numpy as np
from pandas.io.pytables import Table


def find_csv_files(root_path, endswith):

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith),
        recursive=True,
    )

    return files


def truncate_csvs_in_root(root_path, name_of_files_to_trunc, len_threshold):

    files_to_truncate = find_csv_files(root_path, name_of_files_to_trunc)

    df_database_unmod = TableDatabase()
    df_database_dropped = TableDatabase()

    for file in files_to_truncate:
        table = Table(file, len_threshold)
        df_database_unmod.include_table(table, drop_row=False)
        df_database_dropped.include_table(table, drop_row=True)

    print(f"Number of jagged tables before: {df_database_unmod.number_of_jagged_dfs}")
    print(f"Number of jagged tables after: {df_database_dropped.number_of_jagged_dfs}")


class TableDatabase(Table):
    """Don't want to store all this data in one object, rather, just want to update
    its parameters that are is is meta of tables."""

    def __init__(self):
        super(Table, self).__init__()
        self.number_of_jagged_dfs = 0
        self.jagged_dfs = []

    def include_table(self, table: Table, drop_row: bool):
        if drop_row == True:
            table.drop_last_row_df()

        equals_threshold = table.check_if_df_len_equals_thres()
        if equals_threshold is False:
            self.number_of_jagged_dfs += 1
            self.jagged_dfs.append(table.path)


class Table(TableDatabase):
    def __init__(self, df_path, len_threshold):
        super().__init__()
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


def main():
    ROOT_PATH = r"/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData"

    truncate_csvs_in_root(
        ROOT_PATH, name_of_files_to_trunc="all_concat_cells.csv", len_threshold=200
    )


# THIS FILE IS WORKING FOR THE WRONG REASON: IT HAS TO IMPORT A TABLE OBJECT FROM SOMEWHERE
# main()
