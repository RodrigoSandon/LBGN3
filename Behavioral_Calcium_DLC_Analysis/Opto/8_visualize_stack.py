import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import pandas as pd
from statistics import mean
from operator import attrgetter
import seaborn as sns
from scipy import stats
import os
import math
from itertools import combinations

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

def main_gridspec():

    circuit = "BLA_NAcShell"
    t1 = "ArchT"
    t2 = "eYFP"
    session_type = "Choice"
    combo = "Block_Trial_Type_Start_Time_(s)"
    subcombo = "(1.0, 'Free')"
    filename = "avg_all_speeds_z_-5_5savgol_avg.csv"

    treatments = ["ArchT", "eYFP", "ChrimsonR", "mCherry"]
    combos = combinations(treatments, 2)
    for combo in combos:
        print(combo)

    fig = plt.figure(figsize=(15, 8))
    # determines how many rows
    outer = gridspec.GridSpec(1, 1, wspace=0.0, hspace=0.0)

    for idx, subevent in enumerate(range(0,1)):
        # determines how many cols in each row
        inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                        subplot_spec=outer[idx], wspace=0.0, hspace=0.0)

        t1_speed_on_df = pd.read_csv(f"/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis/BetweenMiceAlignmentData/{circuit}/{t1}/{session_type}/{combo}/{subcombo}/{filename}")
        
        t2_speed_on_df = pd.read_csv(f"/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis/BetweenMiceAlignmentData/{circuit}/{t2}/{session_type}/{combo}/{subcombo}/{filename}")
        
        t1_speed_on_df_list = stats.zscore(list(t1_speed_on_df["Avg_Speed_(cm/s)"])) # z scored
        t2_speed_on_df_list = list(t2_speed_on_df["Avg_Speed_(cm/s)"])

        result = stats.pearsonr(t1_speed_on_df_list, t2_speed_on_df_list)
        corr_coef = list(result)[0]
        print(f"{subevent} corr coef: {corr_coef}")
        max_val = round(max(t1_speed_on_df_list), 0)
        min_val = round(min(t1_speed_on_df_list + t2_speed_on_df_list), 0)
        speed_yticks = np.arange(min_val, max_val + 1, 1)

        t = list(t1_speed_on_df["Time_(s)"])
        print(type(t[0]))
        """
        print(t)
        t_neg = np.arange(-5, 0.0, 0.03333)
        t_pos = np.arange(0.0, 5.0, 0.03333)
        t = t_neg.tolist() + t_pos.tolist()
        t = [round(i, 1) for i in t]"""


        ax_1 = plt.Subplot(fig, inner[0])
        ax_1.plot(t, t1_speed_on_df_list, c="green", label=t1)
        #ax_1.plot(t, speed_off_df_list, c="indianred", label="Off")
        ax_1.legend()
        ax_1.set_yticks(speed_yticks)
        ax_1.set_ylim(min_val, max_val + 1)
        ax_1.set_ylabel("Avg. Z-scored Velocity (cm/s)")

        fig.add_subplot(ax_1)

        ax_2 = plt.Subplot(fig, inner[1])
        ax_2.plot(t, t2_speed_on_df_list, label=t2)
        #ax_2.plot(t, speed_off_df_list, c="indianred", label="Off")
        ax_2.legend()
        ax_2.set_yticks(speed_yticks)
        ax_2.set_xticks(t)
        ax_2.set_ylim(min_val, max_val + 1)
        ax_2.set_ylabel("Avg. Z-scored Velocity (cm/s)")
        ax_2.set_xlabel("Time relative to start time (s)")
        ax_2.locator_params(axis='x', nbins=10)

        fig.add_subplot(ax_2)

    # to have proper time period, just insert the -10 to 10 into x axis 
    plt.savefig(f"/media/rory/Padlock_DT/BLA_Analysis/Results/{circuit}_{t1}_v_{t2}_{session_type}_{combo}_{subcombo}_{filename}.png")

def find_subcombos(combo_path):
    subcombos = []
    for i in os.listdir(combo_path):
        subcombos.append(i)
    return subcombos

def main():

    ROOT = r"/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis/BetweenMiceAlignmentData"
    circuits = ["BLA_NAcShell", "vHPC_NAcShell", "vmPFC_NAcShell"]
    treatments = ["ArchT", "eYFP", "ChrimsonR", "mCherry"]
    treatment_colors = {"ArchT": "green", "eYFP" : "gray", "ChrimsonR": "red", "mCherry": "pink"}
    treatment_combinations = combinations(treatments, 2)
    session_type = "Choice"
    combos = ["Block_Trial_Type_Start_Time_(s)"]

    avg_sem_filename = "all_mice_avg_speed_w_sem.csv"

    for circuit in circuits:
        for c in treatment_combinations:
            for combo in combos:
                
                t1 = c[0]
                t2 = c[1]

                subcombos = find_subcombos(f"{ROOT}/{circuit}/{t1}/{session_type}/{combo}")
                for subcombo in subcombos:
                    
                    print(subcombo)

                    try:
                        fig = plt.figure(figsize=(12, 6))
                        # determines how many rows
                        outer = gridspec.GridSpec(1, 1, wspace=0.0, hspace=0.0)

                        for idx, subevent in enumerate(range(0,1)):
                            # determines how many cols in each row
                            inner = gridspec.GridSpecFromSubplotSpec(1, 1,
                                            subplot_spec=outer[idx], wspace=0.0, hspace=0.0)

                            t1_speed_on_df = pd.read_csv(f"{ROOT}/{circuit}/{t1}/{session_type}/{combo}/{subcombo}/{avg_sem_filename}")
                            
                            t2_speed_on_df = pd.read_csv(f"{ROOT}/{circuit}/{t2}/{session_type}/{combo}/{subcombo}/{avg_sem_filename}")
                            
                            # savgol filtered and z-scored

                            t1_speed_on_df_list = list(t1_speed_on_df["Avg_speed_(cm/s)"])
                            t2_speed_on_df_list = list(t2_speed_on_df["Avg_speed_(cm/s)"])

                            t1_sem_speed_on_df_list = list(t1_speed_on_df["SEM"])
                            t2_sem_speed_on_df_list = list(t2_speed_on_df["SEM"])

                            result = stats.pearsonr(t1_speed_on_df_list, t2_speed_on_df_list)
                            corr_coef = list(result)[0]

                            print(f"{circuit}_{t1}_v_{t2}_{session_type}_{combo}_{subcombo}")
                            print(f"corr coef: {corr_coef}")

                            max_val = round(max(t1_speed_on_df_list), 1)
                            min_val = round(min(t1_speed_on_df_list + t2_speed_on_df_list), 1)
                            speed_yticks = np.arange(min_val - 1, max_val + 1, 1)

                            t = list(t1_speed_on_df["Time_(s)"])


                            ax_1 = plt.Subplot(fig, inner[0])

                            ax_1.plot(t, t1_speed_on_df_list, c="black", label=t1)
                            difference_t1_1 = [t1_speed_on_df_list[i] - t1_sem_speed_on_df_list[i] for i, value in enumerate(t1_speed_on_df_list)]
                            difference_t1_2 = [t1_speed_on_df_list[i] + t1_sem_speed_on_df_list[i] for i, value in enumerate(t1_speed_on_df_list)]
                            ax_1.fill_between(t, difference_t1_1, difference_t1_2, facecolor = treatment_colors[t1])

                            ax_1.plot(t, t2_speed_on_df_list,c="black", label=t2)
                            difference_t2_1 = [t2_speed_on_df_list[i] - t2_sem_speed_on_df_list[i] for i, value in enumerate(t2_speed_on_df_list)]
                            difference_t2_2 = [t2_speed_on_df_list[i] + t2_sem_speed_on_df_list[i] for i, value in enumerate(t2_speed_on_df_list)]
                            ax_1.fill_between(t, difference_t2_1, difference_t2_2, facecolor = treatment_colors[t2] )

                            ax_1.legend()
                            ax_1.set_yticks(speed_yticks)
                            ax_1.set_xticks(t)
                            ax_1.set_ylim(min_val - 1, max_val + 1)
                            ax_1.set_ylabel("Avg. Savgol Z-scored Speed (cm/s)")
                            ax_1.set_xlabel("Time relative to start time (s)")
                            ax_1.locator_params(axis='x', nbins=10)
                            ax_1.locator_params(axis='y', nbins=6)
                            fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

                            fig.add_subplot(ax_1)

                        plt.savefig(f"/media/rory/Padlock_DT/Opto_Speed_Analysis/Results/{circuit}_{t1}_v_{t2}_{session_type}_{combo}_{subcombo}_{avg_sem_filename}_v2.png")
                    
                    except ValueError as e:
                        print(e)
                        pass
#main_gridspec()
main()

