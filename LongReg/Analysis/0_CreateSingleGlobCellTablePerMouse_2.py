from cv2 import trace
import os, glob
import pandas as pd
from statistics import mean
from scipy import stats
import math

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

    sessions = ["Pre-RDT RM", "RDT D1"]
    event = "Block_Reward Size_Choice Time (s)"
    subevent = "(2.0, 'Large')"
    dff_file = "plot_ready_z_fullwindow.csv"
    info_df_file = "avg_plot_ready_z_fullwindow.csv"

    glob_cells_d = {

    }

    print("INDV LOCAL CELL TRIALS, SEPARATE GLOBAL CELLS")
    organize_dffs(glob_cells_d, cellreg_files, sessions, event, subevent, dff_file, dff_type="trial")
    # write code to plot each one here w/ gridspec
    for glob_cell, val_1 in glob_cells_d.items():
        print(glob_cell)
        print("########################################")
        for local_cell, val_2 in glob_cells_d[glob_cell].items():
            print(f"|---->{local_cell}")
            for trial, trace in glob_cells_d[glob_cell][local_cell].items():
                print(f"    |---->{trial} : {trace[:1]}")
    
    # reorganize to get sem
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
    
    print("SESSION SEPARATED AVGD & SEM LOCAL CELL TRIALS")
    info_d = get_stats_from_dffs(new_d)
    for session, stat_d in info_d.items():
        print(session)
        print("########################################")
        for stat, stat_lst in info_d[session].items():     
            print(f"|---->{stat} : {stat_lst[:1]}")

    """print("AVGD LOCAL CELL TRIALS, SEPARATE GLOBAL CELLS")
    organize_dffs(glob_cells_d, cellreg_files, sessions, event, subevent, info_df_file, dff_type="avgd")
    # write code to plot each one here w/ gridspec

    for glob_cell, val_1 in glob_cells_d.items():
        print(glob_cell)
        print("########################################")
        for local_cell, val_2 in glob_cells_d[glob_cell].items():
            print(f"|---->{local_cell}")
            for trial, trace in glob_cells_d[glob_cell][local_cell].items():
                print(f"    |---->{trial} : {trace[:1]}")

    print("SESSION SEPARATED AVGD LOCAL CELL TRIALS")
    new_d = re_organize_dffs(glob_cells_d, session_type_1=sessions[0], session_type_2=sessions[1], dff_type="avgd")
    # write code to plot both here w/ gridspec

    for session, val_1 in new_d.items():
        print(session)
        print("########################################")
        for local_cell, val_2 in new_d[session].items():
            print(f"|---->{local_cell} : {new_d[session][local_cell][:1]}")"""


if __name__ == "__main__":
    main()