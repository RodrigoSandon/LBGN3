import pandas as pd
import numpy as np
import os, glob
from typing import List
from pathlib import Path
from csv import writer
from statistics import mean
from operator import attrgetter
from itertools import combinations_with_replacement
from scipy import stats

def find_paths_endswith(root_path, endswith) -> List:

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )

    return files

class Cell:
    def __init__(self,cell_name, dff_trace):
        self.cell_name = cell_name
        self.dff_trace = dff_trace
        self.mean = mean(dff_trace)

def which_batch(mouse: str) -> str:
    mouse_num = mouse.split("-")[2]
    batch = None
    if mouse_num == "1" or mouse_num == "2" or mouse_num == "3":
        batch = "PTP_Inscopix_#1"
    elif mouse_num == "5" or mouse_num == "6" or mouse_num == "7":
        batch = "PTP_Inscopix_#3"
    elif mouse_num == "8" or mouse_num == "9" or mouse_num == "11" or mouse_num == "13":
        batch = "PTP_Inscopix_#4"
    else:
        batch = "PTP_Inscopix_#5"

    return batch

def get_max_of_df(df: pd.DataFrame):
    global_max = 0
    max_vals = list(df.max())

    for i in max_vals:
        if i > global_max:
            global_max = i
 
    return global_max

def get_min_of_df(df: pd.DataFrame):
    global_min = 9999999
    min_vals = list(df.min())

    for i in min_vals:
        if i < global_min:
            global_min = i
 
    return global_min

class Trial:
    def __init__(
        self,
        mouse,
        session,
        event,
        subevent,
        trial_number,
        cell,
        dff_trace,
    ):
        self.mouse = mouse
        self.session = session
        self.event = event
        self.subevent = subevent


        self.trial_number = trial_number
        self.cell = cell
        self.trial_dff_trace = dff_trace

def find_different_subevents(csv_paths: list, subevent_at: int) -> list:
    subevents = []
    curr = None
    for csv in csv_paths:
        part = csv.split("/")[subevent_at]
        if part != curr:
            subevents.append(part)
            curr = part
    
    return subevents

def fill_points_for_hm(df):
    transposed_df = df.transpose()
    #print(df.head())
    #print(transposed_df.head())

    for row in list(transposed_df.columns):
        for col in list(transposed_df.columns):
            if int(transposed_df.loc[row, col]) == 0:
                transposed_df.loc[row, col] = df.loc[row, col]

    return df

def sort_cells(df):
    sorted_cells = []

    for col in list(df.columns):
        cell = Cell(cell_name=col, dff_trace=list(df[col]))
        
        sorted_cells.append(cell)

    sorted_cells.sort(key=attrgetter("mean"), reverse=True)

    def convert_lst_to_d(lst):
        res_dct = {}
        for count, i in enumerate(lst):
            i: Cell
            res_dct[i.cell_name] = i.dff_trace

        print(f"NUMBER OF CELLS: {len(lst)}")
        return res_dct

    sorted_cells_d = convert_lst_to_d(sorted_cells)

    df_mod = pd.DataFrame.from_dict(
        sorted_cells_d
    )  
    # from_records automatically sorts by key
    # from_dict keeps original order intact

    return df_mod

def main():
    ####### Making the pearson corr map #######
    blocks = ["1.0", "2.0", "3.0"]
    rew = ["Large", "Small"]
    shock = ["False", "True"]

    sessions = ["Pre-RDT RM"]

    # Want to generalize results, so include all mice
    for block in blocks:
        for r in rew:
            for s in shock:
                for session in sessions:
                    root = f"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Unnorm_3Bin_Arranged_Dataset_-10_10/{block}/{r}/{s}"
                    mice_files = find_paths_endswith(root, f"{session}/trials_average.csv")
                    #print(mice_files)

                    dst = f"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Unnorm_3Bin_Generalized_PCA_-10_10/{block}/{r}/{s}/{session}"

                    if mice_files:
                        print(f"CURR CSV: {mice_files[0]}")
                        os.makedirs(dst, exist_ok=True)
                        all_cells_df: pd.DataFrame
                        all_cells_df = pd.read_csv(mice_files[0])
                        mouse_num = mice_files[0].split("/")[10].split("-")[2]
                        temp_cols = list(range(0,len(all_cells_df.columns)))
                        all_cells_df = pd.read_csv(mice_files[0], header=None, names=temp_cols)
                        #print(df.head())
                        

                        all_cells_df = all_cells_df.T
                        #df.columns = df.loc[0]
                        all_cells_df.columns = [f"{mouse_num}_{cell_name}" for cell_name in list(all_cells_df.loc[0])]
                        all_cells_df = all_cells_df.iloc[1:, 1:]
                        
                        for csv in mice_files[1:]:
                            
                            print(f"CURR CSV: {csv}")
                            df: pd.DataFrame
                            df = pd.read_csv(csv)
                            mouse_num = csv.split("/")[10].split("-")[2]
                            temp_cols = list(range(0,len(df.columns)))
                            df = pd.read_csv(csv, header=None, names=temp_cols)
                            #print(df.head())
                            

                            df = df.T
                            df.columns = [f"{mouse_num}_{cell_name}" for cell_name in list(df.loc[0])]
                            df = df.iloc[1:, 1:]
                            
                            # add cells to all_cells_df
                            
                            for col in df:
                                nan_exists = False
                                for i in list(df[col]):
                                    if pd.isna(i):
                                        nan_exists=True
                                    
                                if nan_exists == False:
                                    all_cells_df[col] = list(df[col])
                                else:
                                    print(f"nan exists in {col}!")
                            
                            
                        print(all_cells_df.head())
                        # Sort cells
                        df_sorted = sort_cells(all_cells_df)
                        
                        
                        df_sorted.to_csv(os.path.join(dst, "all_cells_avg_trials.csv"))

if __name__ == "__main__":
    main()