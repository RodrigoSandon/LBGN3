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

    def add_trace(self, event, subevent, trace):

        if event in self.traces_d:
                self.traces_d[event][subevent] = trace
        else:
            self.traces_d[event] = {}
            self.traces_d[event][subevent] = trace

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
    dff_file = "avg_plot_ready_z_fullwindow.csv"

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
        df_cellreg = pd.read_csv(file)
        # get len of df_cellreg to know how many global cells there are
        num_glob_cells = len(df_cellreg)

        # or the number of rows (glob cells)
        for count in range(num_glob_cells):
            glob_cell_name = f"glob_cell_{count}"
            glob_cell_obj = GlobalCell(glob_cell_name)

            # Get all of the cells on the same row
            for session in sessions:
                local_cell_name = df_cellreg.loc[count, session]
                glob_cell_obj.local_cells[local_cell_name] = LocalCell(local_cell_name, session)
                local_cell_obj : LocalCell
                local_cell_obj = glob_cell_obj.local_cells[local_cell_name] 
                local_cell_obj.add_trace(event, subevent, [])

            glob_cells_d[glob_cell_name] = glob_cell_obj

        # have your global and local cell  info appropriatly, add traces now
        for glob_cell_name, glob_cesll_obj in glob_cells_d.items():
            #print(glob_cell_name)
            glob_cell_obj : GlobalCell
            glob_cell_obj = glob_cells_d[glob_cell_name]

            for local_cell_name, local_cell_obj in glob_cell_obj.local_cells.items():
                #print("|----> ", local_cell_name)
                local_cell_obj : LocalCell

                # find the trace for this specific cell, session, event, subevent, and trace type
                # remember, trace list is initiated already
                for event in local_cell_obj.traces_d:

                    mouse_root = file.replace("/cellreg_Pre-RDT RM_RDT D1.csv", "")
                    trace_csv_path = f"{mouse_root}/{local_cell_obj.session}/SingleCellAlignmentData/{local_cell_name}/{event}/{subevent}/{dff_file}"
                    trace_df = pd.read_csv(trace_csv_path)
                    trace = list(trace_df.loc[:, local_cell_name])

                    local_cell_obj.add_trace(event, subevent, trace)

                    #print("--------> ", local_cell_obj.session,"|", event,"|", subevent, "|", local_cell_obj.traces_d[event][subevent][:3])

                    # Now have traces in dict for global's local cells - well labelled
                    #time to plot it on gridspec
        # now avg glob cells for this mouse
        """

        avg_glob_cell = {
            session_0: [[trace_0],[trace_1]],
            session_1: [[trace_0],[trace_1]]
        }

        """

        avg_glob_cell = {

        }

        for key, value in glob_cells_d.items():
            glob_cell : GlobalCell
            glob_cell = glob_cells_d[key]
            print(f"{glob_cell.name}")
            for k2, v2, in glob_cell.local_cells.items():
                local_cell : LocalCell
                local_cell = glob_cell.local_cells[k2]
                print(f"|---->{local_cell.name}, {local_cell.session}")
                for session in sessions:
                    if local_cell.session == session:
                        # now only process for event/subevent:
                        # check if session in there
                        if session in avg_glob_cell:
                            avg_glob_cell[session].append(local_cell.traces_d[event][subevent])
                        else :
                            avg_glob_cell[session] = []
                            avg_glob_cell[session].append(local_cell.traces_d[event][subevent])
        
        # all traces should be in there, check, now avg them using zip
        print(avg_glob_cell)

        #example how to avg list of lists column wise, with "a" being lists of lists
        print(*map(mean, zip(*a)))
        break
    """for key, value in glob_cells_d.items():
        glob_cell : GlobalCell
        glob_cell = glob_cells_d[key]
        print(f"{glob_cell.name}")
        for k2, v2, in glob_cell.local_cells.items():
            local_cell : LocalCell
            local_cell = glob_cell.local_cells[k2]
            print(f"|---->{local_cell.name}")
            for k3, v3, in local_cell.traces_d.items():
                print(f"-------->{k3} : {v3}")"""

if __name__ == "__main__":
    main()