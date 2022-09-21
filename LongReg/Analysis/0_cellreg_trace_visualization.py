import os, glob
import pandas as pd

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

    def add_local_cell(self, name, session, event, subevent):
        self.local_cells[name] = LocalCell(name, session, event, subevent)
    

class LocalCell:

    def __init__(self, name, session, event, subevent):
        self.name = name
        self.session = session
        self.event = event
        self.subevent = subevent
        self.traces_d = {} # trace_type : trace 

    def add_trace(self, trace_type, trace):
        self.traces_d[trace_type] = trace

    

def find_file(root_path: str, filename: str) -> list:
    files = glob.glob(
        os.path.join(root_path, "**", filename), recursive=True,
    )
    return files

def main():
    ROOT = r"/media/rory/Padlock_DT/BLA_Analysis"
    """
    We are given the mouse, session and cell_name from the cellreg table (and it's directory)
    We need: event,  subevent, fullwindow zscored dff trace + fullwindow zscored cell identity

    class: GlobalCell
    - a globalcell has it's x local cells
    - each x local cell has it's session i cares about
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
    subevent = "(1.0, 'Large')"
    trace_type = "z_fullwindow"
    dff_file = "avg_plot_ready_z_fullwindow.csv"

    glob_cells_d = {

    }

    for file in cellreg_files:
        df_cellreg = pd.read_csv(file)
        # get len of df_cellreg to know how many global cells there are
        num_glob_cells = len(df_cellreg)

        for count in range(num_glob_cells):
            glob_cell_name = f"cell_{count}"
            glob_cell_obj = GlobalCell(glob_cell_name)

            # Get all of the cells on the same row
            for session in sessions:
                local_cell_name = df_cellreg.loc[count, session]
                glob_cell_obj.add_local_cell(local_cell_name, session, event, subevent)

            glob_cells_d[glob_cell_name] = glob_cell_obj

        # have your global and local cell  info appropriatly, just add traces now
        for glob_cell_name, glob_cesll_obj in glob_cells_d.items():
            print(glob_cell_name)
            glob_cell_obj : GlobalCell
            glob_cell_obj = glob_cells_d[glob_cell_name]
            for local_cell_name, local_cell_obj in glob_cell_obj.local_cells.items():
                local_cell_obj : LocalCell

                # find the trace for this specific cell, session, event, subevent, and trace type
                mouse_root = file.replace("/cellreg_Pre-RDT RM_RDT D1.csv", "")
                trace_csv_path = f"{mouse_root}/{local_cell_obj.session}/SingleCellAlignmentData/{local_cell_name}/{event}/{subevent}/{dff_file}"
                trace_df = pd.read_csv(trace_csv_path)
                trace = list(trace_df.loc[:, local_cell_name])

                local_cell_obj.add_trace(trace_type, trace)

                print(local_cell_name, ":", local_cell_obj.session,"|", local_cell_obj.event,"|", local_cell_obj.subevent, "|", local_cell_obj.traces_d[trace_type][:3])

                # Now have traces in dict for global's local cells - well labelled
                #time to plot it on gridspec

        break

if __name__ == "__main__":
    main()