from cv2 import trace
import os, glob
import pandas as pd
from statistics import mean
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import combinations
import numpy as np
import math
import random

"""
- There's never been a file that indicated what cell needs to correspond to which across session
- But there as been a time where I had to align corresponding cells across different events 
- What is the code I can use again? 
- Just the code for when I have the traces for the longreg
- but I have a database already of all the cells identities and traces,just need to pull it out

So goal is to not make a separate database of longreg, but just using the longreg table to pull out what we need
"""

class GlobalCell:

    def __init__(self, name):
        self.name = name
        self.local_cells = {}
    

class LocalCell:

    def __init__(self, name, session):
        self.name = name
        self.session = session
        self.traces_d = {}

    # add by trial
    def add_trace(self, event, subevent, trial, trace):

        if event in self.traces_d:
            if subevent in self.traces_d[event]:
                self.traces_d[event][subevent][trial] = trace
        else:
            self.traces_d[event] = {}
            self.traces_d[event][subevent] = {}
            self.traces_d[event][subevent][trial] = trace

def find_file(root_path: str, filename: str) -> list:
    files = glob.glob(
        os.path.join(root_path, "**", filename), recursive=True,
    )
    return files

# dicts are mutable so don't have to return anything
def organize_dffs(glob_cells_d: dict, cellreg_files: list, sessions: list, event: str, subevent: str, dff_file: str, dff_type: str):
    dff_name: str
    
    for file in cellreg_files:
        mouse_root = "/".join(file.split("/")[:-1])
        mouse = file.split("/")[6]
        df_cellreg = pd.read_csv(file)
        # get len of df_cellreg to know how many global cells there are
        num_glob_cells = len(df_cellreg)

        for count in list(range(0, num_glob_cells)):
            glob_cell_name = f"{mouse}_glob_cell_{count}"
            # print(glob_cell_name)
            glob_cells_d[glob_cell_name] = {}

            # Get all of the cells on the same row
            for session in sessions:
                # print(count)
                cell = df_cellreg.loc[count, session]
                local_cell_name = f"{mouse}_{session}_{cell}"

                glob_cells_d[glob_cell_name][local_cell_name] = {}

                traces_csv_path = f"{mouse_root}/{session}/SingleCellAlignmentData/{cell}/{event}/{subevent}/{dff_file}"
                traces_df = pd.read_csv(traces_csv_path)

                # do this if it's trials
                if (dff_type == "trial"):
                    traces_df = traces_df.T
                    traces_df = traces_df.iloc[1:]

                for count_2, col in enumerate(traces_df.columns.tolist()):
                    col_list = list(traces_df[col])

                    if (dff_type == "trial"):
                        dff_name = f"Trial_{count_2 + 1}"
                    elif (dff_type == "avgd"):
                        dff_name = "avg_trials_trace"
                    
                    glob_cells_d[glob_cell_name][local_cell_name][dff_name] = col_list

def re_organize_dffs(glob_cells_d: dict, session_type_1: str, session_type_2: str, dff_type: str) -> dict:

    new_d = {
        session_type_1 : {},
        session_type_2 : {}    
    }

    for glob_cell, val_1 in glob_cells_d.items():
        for local_cell, val_2 in glob_cells_d[glob_cell].items():
            if session_type_1 in local_cell:
                if dff_type == "avgd":
                    new_d[session_type_1][local_cell] = list(glob_cells_d[glob_cell][local_cell].values())[0]
                elif dff_type == "trial":
                    new_d[session_type_1][local_cell] = glob_cells_d[glob_cell][local_cell]

            if session_type_2 in local_cell:
                if dff_type == "avgd":
                    new_d[session_type_2][local_cell] = list(glob_cells_d[glob_cell][local_cell].values())[0]
                elif dff_type == "trial":
                    new_d[session_type_2][local_cell] = glob_cells_d[glob_cell][local_cell]
    
    return new_d

# average across local cells
def get_stats_from_dffs(new_d: dict) -> dict:
    info_d = {}

    for session, cells_d in new_d.items():
        if session not in info_d:
             info_d[session] = {}

        count = 0
        lists_of_lists = []
        for local_cell, trials_d in new_d[session].items():
            for trial, trace in new_d[session][local_cell].items():
                lists_of_lists.append(trace)
                count += 1
    
        zipped = zip(*lists_of_lists)
        # print(zipped)

        avg_lst = []
        sem_lst = []

        for index, tuple in enumerate(zipped):
            #print(tuple)
            avg_o_timepoint = sum(list(tuple)) / count
            sem_o_timepoint = stats.tstd(list(tuple))/(math.sqrt(count))
            avg_lst.append(avg_o_timepoint)
            sem_lst.append(sem_o_timepoint)

        info_d[session]["avg"] = avg_lst
        info_d[session]["sem"] = sem_lst
    
    return info_d

# average within local cell
def get_stats_from_dffs_2(new_d: dict) -> dict:
    info_d = {

    }

    for glob_cell, local_cells_d in new_d.items():
        if glob_cell not in info_d:
             info_d[glob_cell] = {}

        count = 0
        lists_of_lists = []
        for local_cell, trials_d in new_d[glob_cell].items():
            for trial, trace in new_d[glob_cell][local_cell].items():
                lists_of_lists.append(trace)
                count += 1
    
            zipped = zip(*lists_of_lists)
            # print(zipped)

            avg_lst = []
            sem_lst = []

            # print(local_cell)
            for index, tuple in enumerate(zipped):
                #print(tuple)
                avg_o_timepoint = sum(list(tuple)) / count
                sem_o_timepoint = stats.tstd(list(tuple))/(math.sqrt(count))
                avg_lst.append(avg_o_timepoint)
                sem_lst.append(sem_o_timepoint)

            info_d[glob_cell][local_cell] = {}
            info_d[glob_cell][local_cell]["avg"] = avg_lst
            info_d[glob_cell][local_cell]["sem"] = sem_lst
    
    return info_d

# average across session across local cell trials
def get_stats_from_dffs_3(new_d: dict) -> dict:
    info_d = {

    }

    for session, local_cells_d in new_d.items():
        if session not in info_d:
             info_d[session] = {}

        count = 0
        lists_of_lists = []
        for local_cell, trials_d in new_d[session].items():
            for trial, trace in new_d[session][local_cell].items():
                lists_of_lists.append(trace)
                count += 1

        print(f"number of trials in {session}: {len(lists_of_lists)}")
        zipped = zip(*lists_of_lists)
        # print(zipped)

        avg_lst = []
        sem_lst = []

        # print(local_cell)
        for index, tuple in enumerate(zipped):
            #print(tuple)
            avg_o_timepoint = sum(list(tuple)) / count
            sem_o_timepoint = stats.tstd(list(tuple))/(math.sqrt(count))
            avg_lst.append(avg_o_timepoint)
            sem_lst.append(sem_o_timepoint)

        info_d[session]["avg"] = avg_lst
        info_d[session]["sem"] = sem_lst
    
    return info_d

def parse_local_cell_name(my_str) -> str:
    bla_num = my_str.split("_")[0].split("-")[-1]
    cell_num = my_str.split("_")[-1]

    result = f"{bla_num}_{cell_num}"
    return result
                    
def main():
    ROOT = r"/media/rory/Padlock_DT/BLA_Analysis"

    cellreg_files = [
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/cellreg_Pre-RDT RM_RDT D1.csv",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/cellreg_Pre-RDT RM_RDT D1.csv",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-13/cellreg_Pre-RDT RM_RDT D1.csv",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-14/cellreg_Pre-RDT RM_RDT D1.csv",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-15/cellreg_Pre-RDT RM_RDT D1.csv",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-16/cellreg_Pre-RDT RM_RDT D1.csv",

    ]
    
    # block 1, averaged 
    sessions = ["Pre-RDT RM", "RDT D1"]
    event = "Block_Reward Size_Choice Time (s)"
    subevent = "(1.0, 'Large')"
    subevent_label = "1.0_Large"
    dff_file = "plot_ready_z_fullwindow.csv"
    info_df_file = "avg_plot_ready_z_fullwindow.csv"

    glob_cells_d = {

    }

    range_from_zero = 10.0
    step = 0.1

    print("INDV LOCAL CELL TRIALS, SEPARATE GLOBAL CELLS")
    organize_dffs(glob_cells_d, cellreg_files, sessions, event, subevent, dff_file, dff_type="trial")
    # write code to plot each one here w/ gridspec
    """for glob_cell, val_1 in glob_cells_d.items():
        print(glob_cell)
        print("########################################")
        for local_cell, val_2 in glob_cells_d[glob_cell].items():
            print(f"|---->{local_cell}")
            for trial, trace in glob_cells_d[glob_cell][local_cell].items():
                print(f"    |---->{trial} : {trace[:1]}")"""
    
    # reorganize to get sem
    """print("SESSION SEPARATED INDV LOCAL CELL TRIALS")
    new_d = re_organize_dffs(glob_cells_d, session_type_1=sessions[0], session_type_2=sessions[1], dff_type="trial")
    # write code to plot both here w/ gridspec

    for session, val_1 in new_d.items():
        print(session)
        print("########################################")
        for local_cell, val_2 in new_d[session].items():
            print(f"|---->{local_cell}")
            for trial, trace in new_d[session][local_cell].items():
                print(f"    |---->{trial} : {trace[:1]}")
    
    print("SESSION SEPARATED AVGD & SEM GLOBAL CELLS")
    info_d = get_stats_from_dffs(new_d)
    for session, stat_d in info_d.items():
        print(session)
        print("########################################")
        for stat, stat_lst in info_d[session].items():     
            print(f"|---->{stat} : {stat_lst[:1]}")"""

    """print("AVGD LOCAL CELL TRIALS, SEPARATE GLOBAL CELLS")
    organize_dffs(glob_cells_d, cellreg_files, sessions, event, subevent, info_df_file, dff_type="avgd")"""
    # write code to plot each one here w/ gridspec

    """for glob_cell, val_1 in glob_cells_d.items():
        print(glob_cell)
        print("########################################")
        for local_cell, val_2 in glob_cells_d[glob_cell].items():
            print(f"|---->{local_cell}")
            for trial, trace in glob_cells_d[glob_cell][local_cell].items():
                print(f"    |---->{trial} : {trace[:1]}")"""
    
    print("GLOBAL CELL SEPARATED AVGD & SEM LOCAL CELL TRIALS")
    info_d = get_stats_from_dffs_2(glob_cells_d)
    glob_cell_count = 0
    for glob_cell, local_cells_d in info_d.items():
        glob_cell_count += 1
        print(glob_cell)
        print("########################################")
        for local_cell, stats_d in info_d[glob_cell].items():
            print(f"|---->{local_cell}")
            for stat, vals in info_d[glob_cell][local_cell].items():    
                print(f"    |---->{stat} : {vals[:1]}")
    print("num global cells: ", glob_cell_count)

    """
    print("SESSION SEPARATED AVGD LOCAL CELL TRIALS")
    new_d = re_organize_dffs(glob_cells_d, session_type_1=sessions[0], session_type_2=sessions[1], dff_type="avgd")
    # write code to plot both here w/ gridspec

    for session, val_1 in new_d.items():
        print(session)
        print("########################################")
        for local_cell, val_2 in new_d[session].items():
            print(f"|---->{local_cell} : {new_d[session][local_cell][:1]}")"""

    """ Need a different label for each global cell"""

    number_of_colors = glob_cell_count

    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
            for i in range(number_of_colors)]

    fig = plt.figure(figsize=(15, 8))
    # determines how many rows, cols
    outer = gridspec.GridSpec(1, 1, wspace=0.0, hspace=0.0)

    for idx, subevent in enumerate(range(0,1)):
        # determines how many cols in each row
        inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                        subplot_spec=outer[idx], wspace=0.0, hspace=0.0)
        
        t_neg = np.arange((-1 * range_from_zero), 0.0, step)
        t_pos = np.arange(0.0, range_from_zero, step)
        t = t_neg.tolist() + t_pos.tolist()
        t = [round(i, 1) for i in t]
        print(t)

        # FIRST SESSION
        ax_1 = plt.Subplot(fig, inner[0])
        # SECOND SESSION
        ax_2 = plt.Subplot(fig, inner[1])

        count = 0
        for glob_cell, local_cells_d in info_d.items():
            for local_cell, stats_d in info_d[glob_cell].items():
                label_name = parse_local_cell_name(local_cell)
                for stat, vals in info_d[glob_cell][local_cell].items():
                    if stat == "avg":
                        if sessions[0] in local_cell:   
                            ax_1.plot(t, vals, color = color[count],label=label_name)
                        if sessions[1] in local_cell:   
                            ax_2.plot(t, vals, color = color[count],label=label_name)
            count += 1

        ax_1.legend()
        ax_1.set_ylabel(f"Avg. Z-scored df/f {sessions[0]}")

        fig.add_subplot(ax_1)

        ax_2.legend()
        ax_2.set_xticks(t)
        ax_2.set_ylabel(f"Avg. Z-scored df/f {sessions[1]}")
        ax_2.set_xlabel("Time relative to start time (s)")
        ax_2.locator_params(axis='x', nbins=10)

        fig.add_subplot(ax_2)

    # plt.show()
    # to have proper time period, just insert the -10 to 10 into x axis 
    fig.savefig(f"/media/rory/Padlock_DT/BLA_Analysis/LongReg/Results/all_glob_cells_avg_local_cells_{event}_{subevent_label}_{sessions[0]}_{sessions[1]}.png")


def main_2():
    ROOT = r"/media/rory/Padlock_DT/BLA_Analysis"

    cellreg_files = [
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/cellreg_Pre-RDT RM_RDT D1.csv",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/cellreg_Pre-RDT RM_RDT D1.csv",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-13/cellreg_Pre-RDT RM_RDT D1.csv",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-14/cellreg_Pre-RDT RM_RDT D1.csv",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-15/cellreg_Pre-RDT RM_RDT D1.csv",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-16/cellreg_Pre-RDT RM_RDT D1.csv",

    ]
    
    # block 1, averaged 
    sessions = ["Pre-RDT RM", "RDT D1"]
    event = "Block_Reward Size_Choice Time (s)"
    subevent = "(1.0, 'Large')"
    subevent_label = "1.0_Large"
    dff_file = "plot_ready_z_fullwindow.csv"
    info_df_file = "avg_plot_ready_z_fullwindow.csv"

    glob_cells_d = {

    }

    range_from_zero = 10.0
    step = 0.1

    print("INDV LOCAL CELL TRIALS, SEPARATE GLOBAL CELLS")
    organize_dffs(glob_cells_d, cellreg_files, sessions, event, subevent, dff_file, dff_type="trial")
    """for glob_cell, val_1 in glob_cells_d.items():
        print(glob_cell)
        print("########################################")
        for local_cell, val_2 in glob_cells_d[glob_cell].items():
            print(f"|---->{local_cell}")
            for trial, trace in glob_cells_d[glob_cell][local_cell].items():
                print(f"    |---->{trial} : {trace[:1]}")"""
    
    print("SESSION SEPARATED INDV LOCAL CELL TRIALS")
    new_d = re_organize_dffs(glob_cells_d, session_type_1=sessions[0], session_type_2=sessions[1], dff_type="trial")
    # write code to plot both here w/ gridspec

    for session, val_1 in new_d.items():
        print(session)
        print("########################################")
        for local_cell, val_2 in new_d[session].items():
            print(f"|---->{local_cell}")
            for trial, trace in new_d[session][local_cell].items():
                print(f"    |---->{trial} : {trace[:1]}")
    
    print("SESSION SEPARATED AVGD & SEM GLOBAL CELLS")
    info_d = get_stats_from_dffs_3(new_d)
    for session, stat_d in info_d.items():
        print(session)
        print("########################################")
        for stat, stat_lst in info_d[session].items():     
            print(f"|---->{stat} : {stat_lst[:1]}")

    session_colors = {sessions[0]: "blue", sessions[1] : "indianred"}

    fig = plt.figure(figsize=(15, 8))
    # determines how many rows, cols
    outer = gridspec.GridSpec(1, 1, wspace=0.0, hspace=0.0)

    for idx, subevent in enumerate(range(0,1)):
        # determines how many cols in each row
        inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                        subplot_spec=outer[idx], wspace=0.0, hspace=0.0)

        result = stats.pearsonr(info_d[sessions[0]]["avg"], info_d[sessions[1]]["avg"])
        corr_coef = list(result)[0]
        print(f"{event} | {subevent} corr coef: {corr_coef}")

        t_neg = np.arange((-1 * range_from_zero), 0.0, step)
        t_pos = np.arange(0.0, range_from_zero, step)
        t = t_neg.tolist() + t_pos.tolist()
        t = [round(i, 1) for i in t]

        # FIRST SESSION
        ax_1 = plt.Subplot(fig, inner[0])
        # SECOND SESSION
        ax_2 = plt.Subplot(fig, inner[1])

        count = 0
        for session, stats_d in info_d.items():
            for stat, vals in info_d[session].items():
                if stat == "avg":
                    if session == sessions[0]:   
                        ax_1.plot(t, vals, color = "black")
                    if session == sessions[1]:   
                        ax_2.plot(t, vals, color = "black")
                elif stat == "sem":
                    difference_1 = [info_d[session]["avg"][idx] - info_d[session]["sem"][idx] for idx,i in enumerate(info_d[session]["avg"])]
                    difference_2 = [info_d[session]["avg"][idx] + info_d[session]["sem"][idx] for idx,i in enumerate(info_d[session]["avg"])]
                    if session == sessions[0]:   
                        facecolor = session_colors[sessions[0]]
                        ax_1.fill_between(t, difference_1, difference_2, facecolor=facecolor)
                    if session == sessions[1]:
                        facecolor = session_colors[sessions[1]] 
                        ax_2.fill_between(t, difference_1, difference_2, facecolor=facecolor)
            count += 1

        ax_1.set_title(f"Avg. Z-scored df/f Across Glob Cells r = {round(corr_coef, 2)}")
        ax_1.set_ylabel(f"{sessions[0]}")

        fig.add_subplot(ax_1)

        ax_2.set_xticks(t)
        ax_2.set_ylabel(f"{sessions[1]}")
        ax_2.set_xlabel("Time relative to start time (s)")
        ax_2.locator_params(axis='x', nbins=10)

        fig.add_subplot(ax_2)

    #plt.show()
    # to have proper time period, just insert the -10 to 10 into x axis 
    fig.savefig(f"/media/rory/Padlock_DT/BLA_Analysis/LongReg/Results/avg_all_glob_cells_avg_local_cells_{event}_{subevent_label}_{sessions[0]}_{sessions[1]}.png")
    

if __name__ == "__main__":
    #main()
    main_2()