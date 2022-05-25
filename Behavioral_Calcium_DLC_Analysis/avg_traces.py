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
    # /media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/RDT D1/Block_Reward Size_Choice Time (s)/(1.0, 'Large')/sorted_traces_z_fullwindow_id_auc_bonf0.05.csv

    ROOT = "/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData"

    files_to_avg = find_paths_endswith(ROOT, "all_concat_cells.csv")

    #len is # of cells
    for file in files_to_avg:
        print(f"CURR FILE: {file}")
        root = Path(file).parent
        df = pd.read_csv(file)
        list_of_traces = []
 
        for col in list(df.columns):
            list_of_traces.append(list(df[col]))

        zipped = zip(*list_of_traces)
        #print(list(zipped))

        #avg = stats.zscore([sum(list(tup))/len(list(tup)) for tup in list(zipped)])

        for tup in list(zipped):
            err = stats.tstd(list(tup))/math.sqrt(len(list(tup)))
            print(err)

        """d = {"Avg dff trace": avg, "SEM": sem}
        avg_df = pd.DataFrame(data=d, index = None)
        avg_df.to_csv(os.path.join(root, "avg_dff_trace_w_sem.csv"), index=False)"""
        break


if __name__ == "__main__":
    main2()
