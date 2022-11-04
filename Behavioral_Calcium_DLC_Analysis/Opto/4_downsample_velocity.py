from typing import List
from pathlib import Path

import os
import glob

import pandas as pd
import matplotlib.pyplot as plt


def find_paths(root_path: Path, endswith: str) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, "**", endswith), recursive=True,
    )
    return files


def main():
    ROOT_PATH = Path(r"/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis")

    avg_filename = "speeds_z_-5_5savgol_avg.csv"

    lst_of_avg_cell_csv_paths = find_paths(ROOT_PATH, avg_filename) #CUSTOMIZE FOR SPECIFIC GROUPINGS YOU WANT TO PROCESS
    
    # check if they need downsampling by opening them and getting length

    for csv in lst_of_avg_cell_csv_paths:
        df = pd.read_csv(csv)
        len_df = len(df)
        # save original as unsampled

        if len_df == 599:
            print(f"{csv} is of length {len_df}")
            old_path = csv.replace(avg_filename, f"{avg_filename}_60fps_unsampled.csv")
            df.to_csv(old_path, index=None)
            # downsampling algo
            df = df[df.index % 2 != 0]

        elif len_df == 1198:
            print(f"{csv} is of length {len_df}")
            old_path = csv.replace(avg_filename, f"{avg_filename}_120fps_unsampled.csv")
            df.to_csv(old_path, index=None)
            #downsampling algo
            df = df[df.index % 4 != 1] # Selects every 4th row starting from 1
        
        print(f"new length: {len(df)}")
        df.to_csv(csv, index=None)
        
def main2():
    ROOT_PATH = Path(r"/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis")
    
    filename = "speeds_z_-5_5savgol_avg.csv"

    avg_filename = f"{filename}_60fps_unsampled.csv"
    lst_of_avg_cell_csv_paths = find_paths(ROOT_PATH, avg_filename) #CUSTOMIZE FOR SPECIFIC GROUPINGS YOU WANT TO PROCESS
    
    # check if they need downsampling by opening them and getting length

    for csv in lst_of_avg_cell_csv_paths:
        df = pd.read_csv(csv)
        len_df = len(df)

        if len_df == 599:
            print(f"{csv} is of length {len_df}")
            # downsampling algo
            df = df.iloc[1::2, :]
        
        print(f"new length: {len(df)}")
        # will turn into new
        df.to_csv(csv.replace("_60fps_unsampled.csv", ""), index=None)

def main3():
    ROOT_PATH = Path(r"/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis")
    
    filename = "speeds_z_-5_5savgol_avg.csv"

    avg_filename = f"{filename}_120fps_unsampled.csv"
    lst_of_avg_cell_csv_paths = find_paths(ROOT_PATH, avg_filename) #CUSTOMIZE FOR SPECIFIC GROUPINGS YOU WANT TO PROCESS
    
    # check if they need downsampling by opening them and getting length

    for csv in lst_of_avg_cell_csv_paths:
        df = pd.read_csv(csv)
        len_df = len(df)

        if len_df == 1198:
            print(f"{csv} is of length {len_df}")
            # downsampling algo
            df = df.iloc[2::4, :]
        
        print(f"new length: {len(df)}")
        # will turn into new
        df.to_csv(csv.replace("_120fps_unsampled.csv", ""), index=None)
        

main3()