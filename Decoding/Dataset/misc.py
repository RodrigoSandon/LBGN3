import matplotlib.pyplot as plt
from typing import List, Optional
import pandas as pd
import seaborn as sns
from operator import attrgetter
from statistics import mean
from itertools import combinations_with_replacement
from scipy import stats
import numpy as np


class Cell:
    def __init__(self,cell_name, dff_trace):
        self.cell_name = cell_name
        self.dff_trace = dff_trace
        self.mean = mean(dff_trace)

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

def hm(
    df,
):
    a = df.T.to_numpy(dtype=float)
    plt.imshow(a, cmap='coolwarm',interpolation='nearest')
    plt.show()

def sort_cells(
    df
):
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
    # print(df_mod)
    return df_mod

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

def fill_points_for_hm(df):
    transposed_df = df.transpose()
    #print(df.head())
    #print(transposed_df.head())

    for row in list(transposed_df.columns):
        for col in list(transposed_df.columns):
            if int(transposed_df.loc[row, col]) == 0:
                transposed_df.loc[row, col] = df.loc[row, col]

    return df

def main():
    example = "/media/rory/Padlock_DT/Scrap/trial_1.csv"
    df : pd.DataFrame
    df = pd.read_csv(example)
    temp_cols = list(range(0,len(df.columns)))
    df = pd.read_csv(example, header=None, names=temp_cols)
    

    df = df.T
    df.columns = df.loc[0]
    df = df.iloc[1:, :]
    print(df.head())

    df_sorted = sort_cells(df)
    print(df_sorted.head())

    cells_list = list(df_sorted.columns)
    

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
        example,
        out_path=example.replace(".csv", "_corrmap.png"),
        vmin=min,
        vmax=max,
        xticklabels=1,
    )
    
    #Save unflattened one-way corrmap
    pearson_corrmap.to_csv(example.replace(".csv", "_corrmap.csv"))


    

if __name__ == "__main__":
    main()