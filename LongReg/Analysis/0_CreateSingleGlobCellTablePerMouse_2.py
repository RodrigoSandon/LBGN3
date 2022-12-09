from cv2 import trace
import os, glob
import pandas as pd
from statistics import mean

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

def make_glob_cells_table(glob_cells_d):
    pass

def main():
    ROOT = r"/media/rory/Padlock_DT/BLA_Analysis"
    """
    We are given the mouse, session and cell_name from the cellreg table (and it's directory)
    We need: event,  subevent, fullwindow zscored dff trace + fullwindow zscored cell identity
 
    class: GlobalCell
    - a globalcell has it's x local cells
    - each x local cell has it's session
    - and each local cell's session has a type of trace we want to analyze from
    """

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

    # a glob cell is a row
    """
    glob_cells_d = {
        glob_cell_0 (name, local_cells_d): {
            local_cell_0 (name, session, traces_d) : {
                traces_d : {
                    event : {
                        subevent : [...traces...]
                    }
                }
            }
        }
    }
    """
    glob_cells_d = {

    }

    for file in cellreg_files:
        mouse_root = "/".join(file.split("/")[:-1])
        mouse = file.split("/")[6]
        print(mouse)
        df_cellreg = pd.read_csv(file)
        # get len of df_cellreg to know how many global cells there are
        num_glob_cells = len(df_cellreg)
        # print(num_glob_cells)
        # print(df_cellreg.head())
        # print(list(range(0, num_glob_cells)))
        # or the number of rows (glob cells)
        for count in list(range(0, num_glob_cells)):
            glob_cell_name = f"{mouse}_glob_cell_{count}"
            print(glob_cell_name)
            glob_cells_d[glob_cell_name] = {}

            # Get all of the cells on the same row
            for session in sessions:
                print(count)
                cell = df_cellreg.loc[count, session]
                local_cell_name = f"{mouse}_{session}_{cell}"
                print(local_cell_name)

                glob_cells_d[glob_cell_name][local_cell_name] = {}

                traces_csv_path = f"{mouse_root}/{session}/SingleCellAlignmentData/{cell}/{event}/{subevent}/{dff_file}"
                traces_df = pd.read_csv(traces_csv_path)

                traces_df = traces_df.T
                traces_df = traces_df.iloc[1:]

                for count_2, col in enumerate(traces_df.columns.tolist()):
                    col_list = list(traces_df[col])
                    trial = f"Trial_{count_2 + 1}"

                    glob_cells_d[glob_cell_name][local_cell_name][trial] = col_list

    for glob_cell, val_1 in glob_cells_d.items():
        print(glob_cell)
        for local_cell, val_2 in glob_cells_d[glob_cell].items():
            print(f"|---->{local_cell}")
            for trial, trace in glob_cells_d[glob_cell][local_cell].items():
                print(f"-------->{trial} : {trace[:1]}")
                

if __name__ == "__main__":
    main()