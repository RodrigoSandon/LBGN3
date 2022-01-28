from TabularDataInterface import TabularDataInterface
import pandas as pd
from typing import Any
import isx
import os

# example dff traces path: /media/rory/RDT VIDS/PTP_Inscopix_#1/BLA-Insc-3/Session-20210120/dff_traces.csv
class ISXTable(TabularDataInterface):
    raw_isx_files_dir = r"/media/rory/PTP Inscopix 1/Inscopix/Raw Inscopix Data Files"
    gpio_path = ""

    """
        Can initiate with having the processed csv file already made 
        and all the methods are just for this purpose.
    
    """

    def __init__(self, path):
        self.path = path
        print("PATH: ", path)
        self.structure_mouse_name = path.split("/")[5]
        self.session_id = path.split("/")[6]
        # print("SESSION ID: ", self.session_id)
        self.root_path = "/".join(
            self.path.split("/")[0 : len(self.path.split("/")) - 1]
        )
        # print("ROOT PATH: ", self.root_path)
        self.processed_isx_csv = self.process_table()

    def __init__(self, path, columns_to_omit):
        self.path = path
        print("PATH: ", path)
        self.structure_mouse_name = path.split("/")[5]
        self.session_id = path.split("/")[6]
        # print("SESSION ID: ", self.session_id)
        self.root_path = "/".join(
            self.path.split("/")[0 : len(self.path.split("/")) - 1]
        )
        # print("ROOT PATH: ", self.root_path)
        self.columns_to_omit = columns_to_omit
        print("COLUMNS TO OMIT: ", columns_to_omit)
        self.processed_isx_csv = self.process_table()

    def csv_to_df(self) -> pd.core.frame.DataFrame:
        df = pd.read_csv(self.path)
        # print(type(df))
        df = df.rename(columns={" ": "Time(s)"})
        # print(df.head())
        return df

    def get_table_type(self) -> Any:
        return str(type(self))

    def find_corresponding_gpio_file(self):

        for root, dirs, files in os.walk(self.raw_isx_files_dir):
            if (
                self.structure_mouse_name
                and "Good Sessions"
                and self.session_id in root
            ):
                for name in files:
                    if name.endswith(".gpio"):
                        self.gpio_path = os.path.join(root, name)

    def export_gpio(self):
        if self.gpio_path == "":
            return "GPIO file hasn't been found yet!"
        else:  # want output csv file to be where it's processed file is
            isx.export_gpio_set_to_csv(
                self.gpio_path,
                str(self.root_path + "/gpio.csv"),
                str(self.root_path),
                time_ref="start",
            )

    def load_gpio_csv(self) -> pd.core.frame.DataFrame:
        gpio_csv_path = ""
        for i in os.listdir(self.root_path):
            if "gpio" in i:
                gpio_csv_path = os.path.join(self.root_path, i)
        if gpio_csv_path == "":
            return "No GPIO csv found!"
        else:
            df = pd.read_csv(gpio_csv_path)
            return df

    def process_table(self, delete_cols=False):
        """
        Not getting rid of time(s) column.
        Will remove indicates columns (by idx).
        Will get rid of rows 0,1, n (n represents frame what which GPIO-1 starts) --> load gpio_csv
        Purpose: to update processed_isx_csv to a processed dataframe

        """
        dff_traces_csv = self.csv_to_df()
        # print("dff_traces: ", dff_traces_csv)
        dff_traces_csv = dff_traces_csv.drop(
            [0]
        )  # drop the first row bc contains strings
        # print(dff_traces_csv.head())

        self.find_corresponding_gpio_file()
        self.export_gpio()

        gpio_csv = self.load_gpio_csv()

        # Find the time for when GPIO-1 is activated
        time_gpio_starts = self.get_time_gpio_starts(gpio_csv)
        print("Time GPIO STARTS: ", time_gpio_starts)

        # Now you go into dff_traces to iterate rows in time column, once find a value greater > stop > get range of rows up to that point and delete them, most efficient?
        processed_isx = self.delete_certain_rows_of_df(dff_traces_csv, time_gpio_starts)
        # print(dff_traces_csv_omitted_rows.head())
        # Now have desired deleted rows and good columns headers, now can delete columns by index
        if delete_cols is False:
            processed_isx = self.delete_certain_cols_of_df(
                processed_isx, self.columns_to_omit
            )
        # print(processed_isx.head())

        return processed_isx

    def get_time_gpio_starts(self, gpio_csv):
        stop_at = 2  # at the second time you see "GPIO-1", that's when you choose the "Time (s)" value
        count = 0
        time_gpio_starts = 0
        # print(gpio_csv.head())
        # print(list(gpio_csv.columns))
        for row in range(len(gpio_csv)):
            if "GPIO-1" in gpio_csv.loc[row, " Channel Name"]:
                count += 1
                if count == stop_at:
                    time_gpio_starts = gpio_csv.loc[row, "Time (s)"]
                    break
        return time_gpio_starts

    def delete_certain_rows_of_df(
        self, dff_traces, remove_rows_with_value_below_this_number
    ) -> pd.core.frame.DataFrame:
        dff_traces = dff_traces.apply(pd.to_numeric)
        dff_traces = dff_traces[
            dff_traces.loc[:, "Time(s)"] >= remove_rows_with_value_below_this_number
        ]
        return dff_traces

    def delete_certain_cols_of_df(
        self, dff_traces, lst_of_cols
    ) -> pd.core.frame.DataFrame:
        cols_to_del = []

        for i in lst_of_cols:  # lst_of_cols are a list of indicies
            col_name = list(dff_traces.columns)[i]
            cols_to_del.append(col_name)

        dff_traces = dff_traces.drop(labels=cols_to_del, axis=1)  # never do inplace
        return dff_traces
