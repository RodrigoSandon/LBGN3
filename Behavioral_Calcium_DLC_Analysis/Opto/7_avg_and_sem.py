import os, glob
from pathlib import Path
import pandas as pd
from scipy import stats
import math

def find_paths_endswith(root_path, endswith) -> list:

    files = glob.glob(
        os.path.join(root_path,"**", "%s") % (endswith), recursive=True,
    )

    return files

def main():

    ROOT = r"/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis/BetweenMiceAlignmentData/"

    files = find_paths_endswith(ROOT, "all_speeds_z_-5_5savgol_renamed.csv")

    for file in files:
        print(f"Currently at: {file}")
        filename = file.split("/")[-1]

        df = pd.read_csv(file)
        list_of_lists = []

        for col in list(df.columns):
            if col != "Time_(s)":
                list_of_lists.append(list(df[col]))

        zipped = zip(*list_of_lists)

        avgs = []
        sems = []

        for tup in list(zipped):
            avg = sum(list(tup)) / len(list(tup))
            avgs.append(avg)
            sem = stats.tstd(list(tup))/(math.sqrt(len(list(tup))))
            sems.append(sem)

        d = {"Time_(s)": list(df["Time_(s)"]),"Avg_speed_(cm/s)": avgs, "SEM": sems}
        df = pd.DataFrame(data=d, index = list(range(0,len(avgs))))
        new_path = file.replace(filename, "all_mice_avg_speed_w_sem.csv")
        
        df.to_csv(new_path, index=False)

def process_one():

    file = "/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis/BetweenMiceAlignmentData/BLA_NAcShell/ArchT/Choice/Block_Trial_Type_Start_Time_(s)/(1.0, 'Free')/all_speeds_z_-5_5savgol_renamed.csv"
    filename = file.split("/")[-1]

    df = pd.read_csv(file)
    list_of_lists = []

    for col in list(df.columns):
        if col != "Time_(s)":
            list_of_lists.append(list(df[col]))

    zipped = zip(*list_of_lists)

    avgs = []
    sems = []

    for tup in list(zipped):
        avg = sum(list(tup)) / len(list(tup))
        avgs.append(avg)
        sem = stats.tstd(list(tup))/(math.sqrt(len(list(tup))))
        sems.append(sem)

    d = {"Avg_speed_(cm/s)": avgs, "SEM": sems}
    df = pd.DataFrame(data=d, index = list(range(0,len(avgs))))
    new_path = file.replace(filename, "all_mice_avg_speed_w_sem.csv")
    
    df.to_csv(new_path, index=False)


if __name__ == "__main__":
    main()
    #process_one()
