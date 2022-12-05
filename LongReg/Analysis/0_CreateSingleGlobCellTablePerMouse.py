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
    subevents = ["(2.0, 'Large')"]
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
        mouse = file.split("/")[6]
        df_cellreg = pd.read_csv(file)
        # get len of df_cellreg to know how many global cells there are
        num_glob_cells = len(df_cellreg)

        # or the number of rows (glob cells)
        for count in range(num_glob_cells):
            glob_cell_name = f"glob_cell_{count}"
            glob_cell_obj = GlobalCell(glob_cell_name)

            # Get all of the cells on the same row
            for session in sessions:
                print(count, session, df_cellreg.loc[count, session])
                local_cell_name = f"{mouse}_{session}_{df_cellreg.loc[count, session]}"
                #print(local_cell_name)
                glob_cell_obj.local_cells[local_cell_name] = LocalCell(local_cell_name, session)
                local_cell_obj : LocalCell
                local_cell_obj = glob_cell_obj.local_cells[local_cell_name] 
                #local_cell_obj.add_trace(event, subevent, [])

            glob_cells_d[glob_cell_name] = glob_cell_obj

        # have your global and local cell  info appropriatly, add traces now
        # don't include a global cell at all if one of it's local cells is missing
        try:
            for glob_cell_name, glob_cell_obj in glob_cells_d.items():
                for local_cell_name, local_cell_obj in glob_cell_obj.local_cells.items():
                    cell = local_cell_name.split("_")[-1]
                    local_cell_obj : LocalCell

                    # find the trace for this specific cell, session, event, subevent, and trace type
                    # remember, trace list is initiated already
                    for subevent in subevents:
                        mouse_root = file.replace("/cellreg_Pre-RDT RM_RDT D1.csv", "")
                        traces_csv_path = f"{mouse_root}/{local_cell_obj.session}/SingleCellAlignmentData/{cell}/{event}/{subevent}/{dff_file}"
                        traces_df = pd.read_csv(traces_csv_path)

                        # Change the name in the middle of processing?
                        # mouse = file.split("/")[6]
                        # full_cell_name = f"{mouse}_{local_cell_name}"
                        # local_cell_obj.name = full_cell_name

                        traces_df = traces_df.T
                        traces_df = traces_df.iloc[1:]

                        for count, col in enumerate(traces_df.columns.tolist()):
                            col_list = list(traces_df[col])[:1]
                            trial = f"Trial_{count + 1}"
                            # print(trial)
                            local_cell_obj.add_trace(event, subevent,trial, col_list)

                        #print("--------> ", local_cell_obj.session,"|", event,"|", subevent, "|", local_cell_obj.traces_d[event][subevent][:3])
        except (FileNotFoundError) as e:
            print(e)
            pass
        """
        avg_glob_cell = {
            session_0: [[trace_0],[trace_1]],
            session_1: [[trace_0],[trace_1]]
        }
        """
        #break
    for key, value in glob_cells_d.items():
        glob_cell : GlobalCell
        glob_cell = glob_cells_d[key]
        print(f"{glob_cell.name}")
        for k2, v2, in glob_cell.local_cells.items():
            local_cell : LocalCell
            local_cell = glob_cell.local_cells[k2]
            print(f"|---->{local_cell.name}")
            # print(f"|-------->{local_cell.traces_d}")
            for k3, v3, in local_cell.traces_d.items():
                # print("here")
                print(f"-------->{k3} : {v3}")
                

if __name__ == "__main__":
    main()


"""avg_glob_cell = {

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
                    avg_glob_cell[session].append(local_cell.traces_d[event][subevent])"""

# all traces should be in there, check, now avg them using zip
# maybe averaging not all neccesary rn, only at the end step when about to visualize
#print(avg_glob_cell)
"""for session, lists_of_lists in avg_glob_cell.items():
    avg = [float(sum(col))/len(col) for col in zip(*lists_of_lists)]
    avg_glob_cell[session] = avg"""