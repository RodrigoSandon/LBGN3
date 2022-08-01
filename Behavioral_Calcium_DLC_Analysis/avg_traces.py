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

def main1():
    # /media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/RDT D1/Block_Reward Size_Choice Time (s)/(1.0, 'Large')/sorted_traces_z_fullwindow_id_auc_bonf0.05.csv

    ROOT = "/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData"

    files_to_avg = find_paths_endswith(ROOT, "sorted_traces_z_fullwindow_id_auc_bonf0.05.csv")

    #len is # of cells
    list_of_traces = []
    for file in files_to_avg:
        print(f"CURR FILE: {file}")
        root = Path(file).parent
        df = pd.read_csv(file)

        zipped = zip(*[list(df["+_mean"]), list(df["-_mean"]), list(df["Neutral_mean"])])

        avg = [(x + y + z)/3 for x,y,z in zipped]

        avg_df = pd.DataFrame(avg, index = None, columns=["Avg dff trace"])
        avg_df.to_csv(os.path.join(root, "avg_dff_trace.csv"), index=False)

def main2():

    file_to_avg = "/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/RDT D1/Block_Reward Size_Choice Time (s)/(3.0, 'Large')/all_concat_cells_z_fullwindow.csv"

    df = pd.read_csv(file_to_avg)
    list_of_traces = []

    for col in list(df.columns):
        list_of_traces.append(list(df[col]))

    zipped = zip(*list_of_traces)

    avgs = []
    sems = []

    for tup in list(zipped):
        avg = sum(list(tup)) / len(list(tup))
        avgs.append(avg)
        sem = stats.tstd(list(tup))/(math.sqrt(len(list(tup))))
        sems.append(sem)

    d = {"Avg dff trace": avgs, "SEM": sems}
    avg_df = pd.DataFrame(data=d, index = list(range(0,len(avgs))))
    avg_df.to_csv(file_to_avg.replace("all_concat_cells_z_fullwindow.csv", "avg_dff_trace_w_sem.csv"), index=False)


if __name__ == "__main__":
    main2()
