import pandas as pd
import numpy as np
import os, glob
from typing import List, Optional
from pathlib import Path
from csv import writer
from itertools import combinations
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

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

def find_paths_mid(root_path: Path, middle: str, endswith: str) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, "**", middle, "**", endswith), recursive=True,
    )
    return files

def find_paths(root_path: Path, endswith: str) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, "**", endswith), recursive=True,
    )
    return files

def find_paths_v2(root_path: Path, endswith: str) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, endswith), recursive=True,
    )
    return files

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
    print(df.head())
    print(transposed_df.head())

    for row in list(transposed_df.columns):
        for col in list(transposed_df.columns):
            if int(transposed_df.loc[row, col]) == 0:
                transposed_df.loc[row, col] = df.loc[row, col]

    return df

def heatmap(
    df,
    file_path,
    out_path,
    cols_to_plot: Optional[List[str]] = None,
    cmap: str = "coolwarm",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    **heatmap_kwargs,
):

    try:
        if cols_to_plot is not None:
            df = df[cols_to_plot]

        ax = sns.heatmap(
            df.transpose(), vmin=vmin, vmax=vmax, cmap=cmap, **heatmap_kwargs
        )
        ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=5)
        ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=5)
        ax.tick_params(left=True, bottom=True)

        plt.title("Pearson Correlation Map")
        plt.savefig(out_path)
        plt.close()

    except ValueError as e:

        print("VALUE ERROR:", e)
        print(f"VALUE ERROR FOR {file_path} --> MAKING CORRMAP")
        pass

def preparation():

    # ex: /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-19/RDT D1/SingleCellAlignmentData/C03/Shock Ocurred_Choice Time (s)/True/plot_ready_z_pre.csv
    ROOT = r"/media/rory/Padlock_DT/BLA_Analysis"

    mice = [
        "BLA-Insc-1",
        "BLA-Insc-2",
        "BLA-Insc-3",
        "BLA-Insc-5",
        "BLA-Insc-6",
        "BLA-Insc-7",
        "BLA-Insc-8",
        "BLA-Insc-9",
        "BLA-Insc-11",
        "BLA-Insc-13",
        "BLA-Insc-14",
        "BLA-Insc-15",
        "BLA-Insc-16",
        "BLA-Insc-18",
        "BLA-Insc-19",
        ]

    session = "RDT D1"

    event = "Shock Ocurred_Choice Time (s)"
    subevents = ["True", "False"]
    # ex of file: /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-13/RDT D1/SingleCellAlignmentData/C05/Shock Ocurred_Choice Time (s)/True/plot_ready_z_fullwindow.csv
    DST_ROOT = "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Pearson_Correlation_Datasets"


    for mouse in mice:
        batch = which_batch(mouse)
            
        """files = find_paths_mid(os.path.join(ROOT, f"{batch}/{mouse}/{session}/SingleCellAlignmentData"), event, "plot_ready_z_fullwindow.csv")

        subevents = find_different_subevents(files, 11)"""

        for subevent in subevents:
            
            files_of_same_group = find_paths(f"{ROOT}/{batch}/{mouse}/{session}/SingleCellAlignmentData",f"{event}/{subevent}/plot_ready_z_fullwindow.csv")

            # create folders whre results will go into
            new_dir = f"{DST_ROOT}/{mouse}/{session}/{event}/{subevent}"
            print(f"Dir being created: {new_dir}")
            os.makedirs(new_dir, exist_ok=True)

            #print(*files_of_same_group, sep="\n")
            for f in files_of_same_group:
                # we are now going through cell csv that are all in same group
                cell = f.split("/")[9]
                df: pd.DataFrame
                df = pd.read_csv(f)
                df = df.iloc[:, 1:]
                # Indicate subwindow you want to decode
                # 0 is at idx 100
                min = 100
                max = 131

                for i in range(len(df)):
                    trial_num = i + 1
                    new_trial = Trial(
                        mouse,
                        session,
                        event,
                        subevent,
                        trial_num,
                        cell,
                        list(df.iloc[i, :])[min:max],
                    )

                    # How the csv will look like: a triangle that are pearson values (not what i have below currently)
                    data = [cell] + new_trial.trial_dff_trace

                    trial_csv_path = os.path.join(new_dir, f"trial_{trial_num}.csv")
                    with open(trial_csv_path, "a") as csv_obj:
                        writer_obj = writer(csv_obj)
                        writer_obj.writerow(data)
                        csv_obj.close()

def make_pearson_corrmaps():
    ####### Making the pearson corr map #######
    DST_ROOT = "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Pearson_Correlation_Datasets"
    mice = [
        "BLA-Insc-1",
        "BLA-Insc-2",
        "BLA-Insc-3",
        "BLA-Insc-5",
        "BLA-Insc-6",
        "BLA-Insc-7",
        "BLA-Insc-8",
        "BLA-Insc-9",
        "BLA-Insc-11",
        "BLA-Insc-13",
        "BLA-Insc-14",
        "BLA-Insc-15",
        "BLA-Insc-16",
        "BLA-Insc-18",
        "BLA-Insc-19",
        ]

    session = "RDT D1"

    event = "Shock Ocurred_Choice Time (s)"
    subevents = ["True", "False"]

    for mouse in mice:

        for subevent in subevents:

            root_dir = f"{DST_ROOT}/{mouse}/{session}/{event}/{subevent}/"
            trial_csvs = find_paths_v2(root_dir,"trial_*.csv")
            #print(trial_csvs)
            df_d = {}
            for csv in trial_csvs:
                print(f"CURR CSV: {csv}")
                df: pd.DataFrame
                df = pd.read_csv(csv)
                temp_cols = list(range(0,len(df.columns)))
                df = pd.read_csv(csv, header=None, names=temp_cols)

                df = df.T
                df.columns = df.loc[0]
                df = df.iloc[1:, :]
                #print(list(df.columns))
                #print(df.head())

                cells_list = list(df.columns)

                combos = list(combinations(cells_list, 2))

                # SETUP SKELETON DATAFRAME
                col_number = len(list(df.columns))
                pearson_corrmap = pd.DataFrame(
                    data=np.zeros((col_number, col_number)),
                    index=list(df.columns),
                    columns=list(df.columns),
                )

                for count, combo in enumerate(combos):
                    #print(f"Working on combo {count}/{len(combos)}: {combo}")

                    cell_x = list(combo)[0]
                    cell_y = list(combo)[1]

                    x = np.array(list(df[cell_x]))
                    y = np.array(list(df[cell_y]))

                    result = stats.pearsonr(x, y)
                    corr_coef = list(result)[0]
                    pval = list(result)[1]

                    pearson_corrmap[cell_x, cell_y] = corr_coef

                #Save plot rdy corrmap
                pearson_corrmap_plt_rdy = fill_points_for_hm(pearson_corrmap)
                heatmap(
                    pearson_corrmap_plt_rdy,
                    csv,
                    out_path=csv.replace(".csv", "_corrmap.png"),
                    vmin=0.5,
                    vmax=0,
                    xticklabels=2,
                )
                
                #Save unflattened one-way corrmap
                pearson_corrmap.to_csv(csv.replace(".csv", "_corrmap.csv"))

                #Save flattened one-way corrmap
                pearson_corrmap_flat = pearson_corrmap.to_numpy().flatten().tolist()
                df_flat = pd.DataFrame(data=pearson_corrmap_flat,index=None,columns=["pearson_corrs"])
                df_flat.to_csv(csv.replace(".csv","_flat_corrmap.csv"), index=False)
                break
            break
        break



if __name__ == "__main__":
    preparation()
    make_pearson_corrmaps()
