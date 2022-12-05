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
    ROOT_PATH = Path(r"/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis_2/")

    filename = "speeds_z_-5_5_savgol.csv"

    csv_paths = find_paths(ROOT_PATH, filename) #CUSTOMIZE FOR SPECIFIC GROUPINGS YOU WANT TO PROCESS
    
    # check if they need downsampling by opening them and getting length
    for csv in csv_paths:
        df = pd.read_csv(csv)
        len_df = len(df.T)
        # save original as unsampled

        if len_df == 600:
            print(f"{csv} is of length {len_df}")
            old_path = csv.replace(filename, f"{filename}_60fps_unsampled.csv")
            df.to_csv(old_path, index=None)
            # downsampling algo
            # df = df.iloc[1::2, :]
            event_col = list(df["Event_#"])
            df = df.iloc[:, 1:]
            df = df.iloc[:, 1::2]
            df.insert(0, column = "Event_#", value=event_col)

        elif len_df == 1199:
            print(f"{csv} is of length {len_df}")
            old_path = csv.replace(filename, f"{filename}_120fps_unsampled.csv")
            df.to_csv(old_path, index=None)
            #downsampling algo
            # df = df.iloc[2::4, :]
            event_col = list(df["Event_#"])
            df = df.iloc[:, 1:]
            df = df.iloc[:, 2::4]
            df.insert(0, column = "Event_#", value=event_col)
        
        print(f"new length: {len(df.T)}")

        df.to_csv(csv, index=None)

def main_avg():
    ROOT_PATH = Path(r"/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis_2/")

    filename = "speeds_z_-5_5_savgol_avg.csv"

    csv_paths = find_paths(ROOT_PATH, filename) #CUSTOMIZE FOR SPECIFIC GROUPINGS YOU WANT TO PROCESS
    
    # check if they need downsampling by opening them and getting length
    for csv in csv_paths:
        df = pd.read_csv(csv)
        len_df = len(df)
        # save original as unsampled

        if len_df == 599:
            print(f"{csv} is of length {len_df}")
            old_path = csv.replace(filename, f"{filename}_60fps_unsampled.csv")
            df.to_csv(old_path, index=None)
            # downsampling algo
            df = df.iloc[1::2, :]
            #df = df.iloc[:, 1:]
            #df = df.iloc[:, 1::2]

        elif len_df == 1198:
            print(f"{csv} is of length {len_df}")
            old_path = csv.replace(filename, f"{filename}_120fps_unsampled.csv")
            df.to_csv(old_path, index=None)
            #downsampling algo
            df = df.iloc[2::4, :]
            #df = df.iloc[:, 1:]
            #df = df.iloc[:, 2::4]
        
        print(f"new length: {len(df)}")

        df.to_csv(csv, index=None)
        
def main2():
    ROOT_PATH = Path(r"/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis")
    
    filename = "speeds_z_-5_5savgol_avg.csv"

    new_unsampled_filename = f"{filename}_60fps_unsampled.csv"
    csv_paths = find_paths(ROOT_PATH, new_unsampled_filename) #CUSTOMIZE FOR SPECIFIC GROUPINGS YOU WANT TO PROCESS
    
    # check if they need downsampling by opening them and getting length

    for csv in csv_paths:
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
        
def one():
    filename = "speeds_z_-5_5_savgol.csv"
    csv = f"/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis_2/BLA_NAcShell/eYFP/Choice/RRD76/body/RRD76_choice_AlignmentData/Block_Trial_Type_Start_Time_(s)/(1.0, 'Free')/{filename}"
    # print(csv)
    df = pd.read_csv(csv)
    # ACTIVATE IF DOING TRIALS, ELSE COMMENT OUT
    len_df = len(df.T)
    # save original as unsampled

    if len_df == 600:
        print(f"{csv} is of length {len_df}")
        old_path = csv.replace(filename, f"{filename}_60fps_unsampled.csv")
        df.to_csv(old_path, index=None)
        # downsampling algo
        # df = df.iloc[1::2, :]
        df = df.iloc[:, 1:]
        df = df.iloc[:, 1::2]

    elif len_df == 1199:
        print(f"{csv} is of length {len_df}")
        old_path = csv.replace(filename, f"{filename}_120fps_unsampled.csv")
        df.to_csv(old_path, index=None)
        #downsampling algo
        # df = df.iloc[2::4, :]
        df = df.iloc[:, 1:]
        df = df.iloc[:, 2::4]
    
    print(f"new length: {len(df.T)}")

    df.to_csv(csv, index=None)

def one_avg():
    filename = "speeds_z_-5_5_savgol_avg.csv"
    csv = f"/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis_2/BLA_NAcShell/eYFP/Choice/RRD17/body/RRD17_choice_AlignmentData/Block_Trial_Type_Start_Time_(s)/(1.0, 'Free')/{filename}"
    # print(csv)
    df = pd.read_csv(csv)
    # ACTIVATE IF DOING TRIALS, ELSE COMMENT OUT
    len_df = len(df)
    # save original as unsampled

    if len_df == 599:
        print(f"{csv} is of length {len_df}")
        old_path = csv.replace(filename, f"{filename}_60fps_unsampled.csv")
        df.to_csv(old_path, index=None)
        # downsampling algo
        df = df.iloc[1::2, :]
        #df = df.iloc[:, 1:]
        #df = df.iloc[:, 1::2]

    elif len_df == 1198:
        print(f"{csv} is of length {len_df}")
        old_path = csv.replace(filename, f"{filename}_120fps_unsampled.csv")
        df.to_csv(old_path, index=None)
        #downsampling algo
        df = df.iloc[2::4, :]
        #df = df.iloc[:, 1:]
        #df = df.iloc[:, 2::4]
    
    print(f"new length: {len(df)}")

    df.to_csv(csv, index=None)

#main()
#one()
main()
main_avg()