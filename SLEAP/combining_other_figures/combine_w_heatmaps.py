import matplotlib.pyplot as plt
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

hm_df = pd.read_csv("/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/RDT D1/Block_Reward Size_Choice Time (s)/(1.0, 'Large')/all_concat_cells_z_fullwindow.csv")

hm_df_sort = sort_cells(hm_df)


speed_df = pd.read_csv("/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/RDT D1/Block_Reward Size_Choice Time (s)/(1.0, 'Large')/avg_unnorm_speed.csv")


t_pos = np.arange(0.0, 10.0, 0.1)
t_neg = np.arange(-10.0, 0.0, 0.1)
t = t_neg.tolist() + t_pos.tolist()
t_ticks = np.arange(0, len(t), 1)

fig, axs = plt.subplots(2, 1, sharex=True)
# Remove horizontal space between axes
fig.subplots_adjust(hspace=0)

# Plot each graph, and manually set the y tick values
hm_yticks = list(hm_df_sort.columns)
#hm_yticks = [i.replace("BLA-Insc-","") for i in hm_yticks]

speed_list = list(speed_df["Avg. Speed (cm/s)"])
speed_max = round(max(list(speed_df["Avg. Speed (cm/s)"])), 1)
speed_min = round(min(list(speed_df["Avg. Speed (cm/s)"])), 1)
speed_yticks = np.arange(speed_min, speed_max, 1.5)

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

hm_df_matrix = hm_df_sort.to_numpy().T

axs[0].pcolormesh(hm_df_matrix, vmin = get_min_of_df(hm_df_sort), vmax = get_max_of_df(hm_df_sort), cmap="bwr")
axs[0].set_yticks(range(0, len(hm_yticks)), hm_yticks)
axs[0].set_ylabel("Neuron #")
axs[0].set_ylim(0, len(hm_yticks))

axs[1].plot(t_ticks, speed_list)
axs[1].set_yticks(speed_yticks)
axs[1].set_ylim(0, speed_max + 1)
axs[1].set_ylabel("Speed (cm/s)")
axs[1].set_xlabel("Time Relative to Choice (s)")

########################################################################

hm_df_2 = pd.read_csv("/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/RDT D1/Block_Reward Size_Choice Time (s)/(2.0, 'Large')/all_concat_cells_z_fullwindow.csv")
hm_df_sort_2 = sort_cells(hm_df_2)

speed_df_2 = pd.read_csv("/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/RDT D1/Block_Reward Size_Choice Time (s)/(2.0, 'Large')/avg_unnorm_speed.csv")

t_pos_2 = np.arange(0.0, 10.0, 0.1)
t_neg_2 = np.arange(-10.0, 0.0, 0.1)
t_2 = t_neg_2.tolist() + t_pos_2.tolist()
t_ticks_2 = np.arange(0, len(t), 1)

fig_2, axs_2 = plt.subplots(2, 1, sharex=True)
# Remove horizontal space between axes
fig_2.subplots_adjust(hspace=0)

# Plot each graph, and manually set the y tick values
hm_yticks_2 = list(hm_df_sort_2.columns)
#hm_yticks = [i.replace("BLA-Insc-","") for i in hm_yticks]

speed_list_2 = list(speed_df_2["Avg. Speed (cm/s)"])
speed_max_2 = round(max(list(speed_df_2["Avg. Speed (cm/s)"])), 1)
speed_min_2 = round(min(list(speed_df_2["Avg. Speed (cm/s)"])), 1)
speed_yticks_2 = np.arange(speed_min_2, speed_max_2, 1.5)

hm_df_matrix_2 = hm_df_sort_2.to_numpy().T
axs_2[0].pcolormesh(hm_df_matrix_2, vmin = get_min_of_df(hm_df_sort_2), vmax = get_max_of_df(hm_df_sort_2), cmap="bwr")
axs_2[0].set_yticks(range(0, len(hm_yticks_2)), hm_yticks_2)
axs_2[0].set_ylabel("Neuron #")
axs_2[0].set_ylim(0, len(hm_yticks_2))

axs_2[1].plot(t_ticks_2, speed_list_2)
axs_2[1].set_yticks(speed_yticks_2)
axs_2[1].set_ylim(0, speed_max_2 + 1)
axs_2[1].set_ylabel("Speed (cm/s)")
axs_2[1].set_xlabel("Time Relative to Choice (s)")

########################################################################

# Create new figure and two subplots, sharing both axes
fig_3, (ax3, ax4) = plt.subplots(1,2,sharey=True, sharex=True)

# Plot data from fig1 and fig2
ax3 = fig
ax4 = fig_2
plt.show()

plt.show()