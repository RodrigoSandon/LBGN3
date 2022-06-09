import os
import glob
from typing import List
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy import stats
import numpy as np

"""Goal: to resample based on before choice time and after choice time, not including the timepoint choice time"""

def avg_cell_eventrace_w_resampling(df: pd.DataFrame, csv_path, cell_name, x_ticks, plot: bool, export_avg: bool):
    """Plots the figure from the csv file given"""
    path_to_save = csv_path.replace("plot_ready_choice_aligned_resampled.csv", "avg_plot_choice_aligned_resampled.png")
    df = df.iloc[:, 1:]
    xaxis = list(df.columns)
    xaxis_len = len(xaxis)

    row_count = len(df)

    # averages columns (timepoints) while skipping NaNs
    avg_of_col_lst = list(df.mean(skipna=True))

    if plot == True:
        fig, ax = plt.subplots()
        every_nth = 20
        ax.plot(xaxis, avg_of_col_lst)
        ax.set_xticks(x_ticks)
        ax.set_xlabel("Time relative to choice (s)")
        for n, label in enumerate(ax.xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)
        for n, label in enumerate(ax.xaxis.get_major_ticks()):
            if n % every_nth != 0:
                label.set_visible(False)
        ax.set_ylabel("Resampled Average DF/F (n=%s)" % (row_count))
        fig.savefig(path_to_save)
        plt.close(fig)

    if export_avg == True:
        path_to_save = csv_path.replace(
            "plot_ready_choice_aligned_resampled.csv", "avg_plot_ready_choice_aligned_resampled.csv")
        export_avg_cell_eventraces(cell_name, avg_of_col_lst, path_to_save)

def export_avg_cell_eventraces(
    cell_name, avg_dff_list_for_timewindow_n_event, out_path
):
    df = pd.DataFrame(avg_dff_list_for_timewindow_n_event, columns=[cell_name])
    df.to_csv(out_path, index=False)


csv = "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-9/RDT D1/SingleCellAlignmentData/C01/Shock Ocurred_Start Time (s)_Collection Time (s)/True/plot_ready_choice_aligned.csv"
cell_name = csv.split("/")[9]
df = pd.read_csv(csv)
col_to_save = list(df.iloc[:, 0])
#print(col_to_save)
df = df.iloc[: , 1:]
#print(df.head())
df = df.T
#print(list(df.index))
desired_start_choice_len = 100
desired_choice_end_len =100
# Total len will be 200 at the end
t_pos = np.arange(0.0, 10.1, 0.1)
t_neg = np.arange(-10.0, 0.0, 0.1)
t = t_neg.tolist() +  t_pos.tolist()
t = [round(i, 1) for i in t]
#print(len(t))

d = {}

# temporarily have col represent all trials so u can resample them
for col in list(df.columns):
 
    trial_start_to_choice = [i for i in df.iloc[:df.index.get_loc("Choice Time"), col] if pd.isna(i) == False]
    trial_choice_to_end = [i for i in df.iloc[df.index.get_loc("Choice Time") + 1:, col] if pd.isna(i) == False]

    resampled_start_to_choice = signal.resample(trial_start_to_choice, desired_start_choice_len)
    resampled_choice_to_end = signal.resample(trial_choice_to_end, desired_choice_end_len)

    final_resampled = list(resampled_start_to_choice) + [df.iloc[df.index.get_loc("Choice Time"), col]] + list(resampled_choice_to_end)
    
    d[col] = final_resampled

new_df = pd.DataFrame.from_dict(d)
new_df = new_df.T
new_df.columns = t
new_df.insert(0,"Event #", col_to_save)
new_csv_path = csv.replace("plot_ready_choice_aligned.csv", "plot_ready_choice_aligned_resampled.csv")
new_df.to_csv(new_csv_path, index=False)

print(new_df.head())

avg_cell_eventrace_w_resampling(new_df, new_csv_path, cell_name, t, plot=True, export_avg=True)

    
