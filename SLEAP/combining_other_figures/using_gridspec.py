import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import pandas as pd
from statistics import mean
from operator import attrgetter
import seaborn as sns

class Cell:
    def __init__(self,cell_name, dff_trace: list):
        self.cell_name = cell_name
        self.dff_trace = dff_trace
        self.mean = mean(dff_trace[100: 131]) 
        #pick which time range to get mean

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
        return res_dct

    sorted_cells_d = convert_lst_to_d(sorted_cells)

    df_mod = pd.DataFrame.from_dict(
        sorted_cells_d
    )  

    return df_mod[df_mod.columns[::-1]] #reversed df
# doing a heatmap as one of the subplots
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

hm_df_2 = pd.read_csv("/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/RDT D1/Block_Reward Size_Choice Time (s)/(2.0, 'Large')/all_concat_cells_z_fullwindow.csv")
hm_df_sort_2 = sort_cells(hm_df_2)
speed_df_2 = pd.read_csv("/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/RDT D1/Block_Reward Size_Choice Time (s)/(2.0, 'Large')/avg_unnorm_speed.csv")
hm_yticks_2 = list(hm_df_sort_2.columns)
speed_list_2 = list(speed_df_2["Avg. Speed (cm/s)"])
speed_max_2 = round(max(list(speed_df_2["Avg. Speed (cm/s)"])), 1)
speed_min_2 = round(min(list(speed_df_2["Avg. Speed (cm/s)"])), 1)
speed_yticks_2 = np.arange(speed_min_2, speed_max_2, 1.5)
hm_df_matrix_2 = hm_df_sort_2.to_numpy().T

hm_df = pd.read_csv("/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/RDT D1/Block_Reward Size_Choice Time (s)/(1.0, 'Large')/all_concat_cells_z_fullwindow.csv")
hm_df_sort = sort_cells(hm_df)
speed_df = pd.read_csv("/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/RDT D1/Block_Reward Size_Choice Time (s)/(1.0, 'Large')/avg_unnorm_speed.csv")
hm_yticks = list(hm_df_sort.columns)
speed_list = list(speed_df["Avg. Speed (cm/s)"])
speed_max = round(max(list(speed_df["Avg. Speed (cm/s)"])), 1)
speed_min = round(min(list(speed_df["Avg. Speed (cm/s)"])), 1)
speed_yticks = np.arange(speed_min, speed_max, 1.5)
hm_df_matrix = hm_df_sort.to_numpy().T
t_pos = np.arange(0.0, 10.0, 0.1)
t_neg = np.arange(-10.0, 0.0, 0.1)
t = t_neg.tolist() + t_pos.tolist()
t_ticks = np.arange(0, len(t), 1)


subevents = ["(1.0, 'Large')", "(2.0, 'Large')", "(3.0, 'Large')"]

fig = plt.figure(figsize=(15, 8))
outer = gridspec.GridSpec(1, len(subevents), wspace=0.0, hspace=0.0)

#across different subevents (horizontally)
for idx, subevent in enumerate(subevents):
    inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                    subplot_spec=outer[idx], wspace=0.0, hspace=0.0)
    # of the same subevent
    hm_df = pd.read_csv(f"/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/RDT D1/Block_Reward Size_Choice Time (s)/{subevent}/all_concat_cells_z_fullwindow.csv")
    speed_df = pd.read_csv(f"/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/RDT D1/Block_Reward Size_Choice Time (s)/{subevent}/avg_unnorm_speed.csv")
    hm_df_sort = sort_cells(hm_df)
    hm_yticks = list(hm_df_sort.columns)
    speed_list = list(speed_df["Avg. Speed (cm/s)"])
    speed_max = round(max(list(speed_df["Avg. Speed (cm/s)"])), 1)
    speed_min = round(min(list(speed_df["Avg. Speed (cm/s)"])), 1)
    speed_yticks = np.arange(speed_min, speed_max, 1.5)
    hm_df_matrix = hm_df_sort.to_numpy().T
    t_pos = np.arange(0.0, 10.0, 0.1)
    t_neg = np.arange(-10.0, 0.0, 0.1)
    t = t_neg.tolist() + t_pos.tolist()
    t_ticks = np.arange(0, len(t), 1)

    if idx > 0:
        ax_1 = plt.Subplot(fig, inner[0])
        ax_1.pcolormesh(hm_df_matrix, vmin = get_min_of_df(hm_df_sort), vmax = get_max_of_df(hm_df_sort), cmap="bwr")
        ax_1.set_yticks([])
        ax_1.set_xticks([])
        ax_1.set_ylim(0, len(hm_yticks))

        fig.add_subplot(ax_1)

        ax_2 = plt.Subplot(fig, inner[1])
        ax_2.plot(t_ticks, speed_list)
        ax_2.set_yticks([])
        ax_2.set_ylim(0, speed_max + 1)
        ax_2.set_xlabel("Time Relative to Choice (s)")

        fig.add_subplot(ax_2)
    else:   
        ax_1 = plt.Subplot(fig, inner[0])
        ax_1.pcolormesh(hm_df_matrix, vmin = get_min_of_df(hm_df_sort), vmax = get_max_of_df(hm_df_sort), cmap="bwr")
        ax_1.set_yticks(range(0, len(hm_yticks)), hm_yticks)
        ax_1.set_xticks([])
        ax_1.set_ylabel("Neuron #")
        ax_1.set_ylim(0, len(hm_yticks))

        fig.add_subplot(ax_1)

        ax_2 = plt.Subplot(fig, inner[1])
        ax_2.plot(t_ticks, speed_list)
        ax_2.set_yticks(speed_yticks)
        ax_2.set_ylim(0, speed_max + 1)
        ax_2.set_ylabel("Speed (cm/s)")
        ax_2.set_xlabel("Time Relative to Choice (s)")

        fig.add_subplot(ax_2)

plt.show()