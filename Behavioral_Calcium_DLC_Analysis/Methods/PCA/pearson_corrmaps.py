import pandas as pd
import numpy as np
import os, glob
from typing import List, Optional
from pathlib import Path
from csv import writer
from statistics import mean
from operator import attrgetter
from itertools import combinations_with_replacement
from scipy import stats
import matplotlib.pyplot as plt
import seaborn  as sns

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
        ax.tick_params(left=True, top=True, labeltop = True, bottom=False, labelbottom=False)

        plt.title("Pearson Correlation Map")
        plt.savefig(out_path)
        plt.close()

    except ValueError as e:

        print("VALUE ERROR:", e)
        print(f"VALUE ERROR FOR {file_path} --> MAKING CORRMAP")
        pass

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

def make_pearson_corrmaps():
    ####### Making the pearson corr map #######
    blocks = ["1.0", "2.0", "3.0"]
    rew = ["Large", "Small"]
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
        "BLA-Insc-19"
    ]

    sessions = ["RDT D1", "RDT D2", "RDT D3"]

    # Want to generalize results, so include all mice
    for block in blocks:
        for r in rew:
            for session in sessions:
                root = f"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Arranged_Dataset/{block}/{r}"
                mice_files = find_paths_endswith(root, f"{session}/trials_average.csv")
                #print(mice_files)

                dst = f"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Generalized_PCA/{block}/{r}/{session}"
                os.makedirs(dst, exist_ok=True)

                print(f"CURR CSV: {mice_files[0]}")
                all_cells_df: pd.DataFrame
                all_cells_df = pd.read_csv(mice_files[0])
                mouse_num = mice_files[0].split("/")[9].split("-")[2]
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
                    mouse_num = csv.split("/")[9].split("-")[2]
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

                cells_list = list(df_sorted.columns)
                #print(cells_list)

                combos = list(combinations_with_replacement(cells_list, 2))
                #print(combos)

                # SETUP SKELETON DATAFRAME
                col_number = len(list(df_sorted.columns))
                pearson_corrmap = pd.DataFrame(
                    data=np.zeros((col_number, col_number)),
                    index=list(df_sorted.columns),
                    columns=list(df_sorted.columns),
                )

                for count, combo in enumerate(combos):
                    #print(f"Working on combo {count}/{len(combos)}: {combo}")

                    cell_x = list(combo)[0]
                    cell_y = list(combo)[1]
                    #print(cell_x)

                    # 4/4/22: why is getting the list from this df so weird (as below)?
                    # i actually don't know, but it must be bc of the prior editing i did
                    # either way, it works - think it's bc we double ran it - bc error when not double runned
                    x = list(df_sorted[cell_x])
                    y = list(df_sorted[cell_y])
                    #print(x)

                    result = stats.pearsonr(x, y)
                    corr_coef = list(result)[0]
                    pval = list(result)[1]

                    #print(corr_coef)
                    pearson_corrmap.loc[cell_x, cell_y] = corr_coef

                #Save plot rdy corrmap
                pearson_corrmap_plt_rdy = fill_points_for_hm(pearson_corrmap)
                #print(pearson_corrmap_plt_rdy)

                max = get_max_of_df(pearson_corrmap_plt_rdy)
                min = get_min_of_df(pearson_corrmap_plt_rdy)

                heatmap(
                    pearson_corrmap_plt_rdy,
                    csv,
                    out_path=os.path.join(dst,"all_cells_avg_trials_corrmap.png"),
                    vmin=min,
                    vmax=max,
                    xticklabels=1,
                )
                
                
                pearson_corrmap_plt_rdy.to_csv(os.path.join(dst, "all_cells_avg_trials_corrmap.csv"))

if __name__ == "__main__":
    # CAN ONLY RUN PREPARATION() ONCE OR ELSE WE GET DOUBLE THE CELLS IN EACH CSV
    #preparation()
    make_pearson_corrmaps()
