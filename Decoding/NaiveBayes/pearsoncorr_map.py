import pandas as pd
import numpy as np
import os, glob
from typing import List
from pathlib import Path

def which_batch(mouse: str) -> str:
    mouse_num = mouse.split("-")[2]
    batch = None
    if mouse_num == "1" or mouse_num == "2" or mouse_num == "3":
        batch = "PTP_Inscopix_#1"
    elif mouse_num == "5" or mouse_num == "6" or mouse_num == "7":
        batch = "PTP_Inscopix_#3"
    elif mouse_num == "8" or mouse_num == "9" or mouse_num == "11" or mouse_num == "13":
        batch = "PTP_Inscopix_#4"
    else:
        batch = "PTP_Inscopix_#5"

    return batch

def find_paths_mid(root_path: Path, middle: str, endswith: str) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, "**", middle, "**", endswith), recursive=True,
    )
    return files

def find_paths(root_path: Path, endswith: str) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, "**", endswith), recursive=True,
    )
    return files

def custom_dissect_path(path: Path):
    parts = list(path.parts)
    event_category = parts[10]
    block, outcome = strip_outcome(parts[11])
    mouse = parts[6]
    session = parts[7]
    cell = parts[9]
    # Now you actually need to go into the the csv file to get the events you want.
    # Do this in another process
    return block, event_category, outcome, mouse, session, cell

class Trial:
    def __init__(
        self,
        block,
        event_category,
        outcome,
        mouse,
        session,
        trial_number,
        cell,
        dff_trace,
        timepoints,
        idx_at_time_zero,
    ):
        self.block = block
        self.event_category = event_category
        self.outcome = outcome
        self.mouse = mouse
        self.session = session
        self.trial_number = trial_number
        self.cell = cell
        self.trial_dff_trace = dff_trace
        self.timepoints = timepoints
        self.idx_at_time_zero = idx_at_time_zero

        self.prechoice_dff_trace = self.get_prechoice_dff_trace()
        self.postchoice_dff_trace = self.get_postchoice_dff_trace()

        self.prechoice_timepoints = self.get_prechoice_timepoints()
        self.postchoice_timepoints = self.get_postchoice_timepoints()

        def get_prechoice_dff_trace(self):
            """Gives you activity at beginning (-10s) to 0"""
            return self.trial_dff_trace[0 : self.idx_at_time_zero + 1]

        def get_postchoice_dff_trace(self):
            """Gives you activity at 0 to 10s"""
            return self.trial_dff_trace[self.idx_at_time_zero + 1 : -1]

        def get_prechoice_timepoints(self):
            return self.timepoints[0 : self.idx_at_time_zero + 1]

        def get_postchoice_timepoints(self):
            return self.timepoints[self.idx_at_time_zero + 1 : -1]

def main():

    # ex: /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-19/RDT D1/SingleCellAlignmentData/C03/Shock Ocurred_Choice Time (s)/True/plot_ready_z_pre.csv
    ROOT = r"/media/rory/Padlock_DT/BLA_Analysis"

    mice = [
        "BLA-Insc-1",
        "BLA-Insc-2",
        "BLA-Insc-3",
        "BLA-Insc-5",
        "BLA-Insc-6",
        "BLA-Insc-7",
        "BLA-Insc-8",
        "BLA-Insc-9",
        "BLA-Insc-11",
        "BLA-Insc-13",
        "BLA-Insc-14",
        "BLA-Insc-15",
        "BLA-Insc-16",
        "BLA-Insc-18",
        "BLA-Insc-19",
        ]

    sessions = ["RDT D1"]

    event = "Shock Ocurred_Choice Time (s)"
    # ex of file: /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-13/RDT D1/SingleCellAlignmentData/C05/Shock Ocurred_Choice Time (s)/True/plot_ready_z_fullwindow.csv

    for mouse in mice:
        batch = which_batch(mouse)
        for session in sessions:
        
            files = find_paths_mid(os.path.join(ROOT, f"{batch}/{mouse}/{session}/SingleCellAlignmentData"), event, "plot_ready_z_fullwindow.csv")

            for csv_path in files:
            df: pd.DataFrame
            df = pd.read_csv(csv_path)
            df = df.iloc[:, 1:]


            for i in range(len(df)):
                trial_num = i + 1
                new_trial = Trial(
                    block,
                    event_category,
                    outcome,
                    mouse,
                    session,
                    trial_num,
                    cell,
                    list(df.iloc[i, :]),
                    timepoints,
                    idx_at_time_zero,
                )

                trial_csv_name = os.path.join(new_dirs, f"trial_{trial_num}.csv")
                header = ["Cell"] + new_trial.get_prechoice_timepoints()
                data = [cell] + new_trial.get_prechoice_dff_trace()

                # look if the csv for this trial exists already
                if os.path.exists(trial_csv_name) == True:
                    with open(trial_csv_name, "a") as csv_obj:
                        writer_obj = writer(csv_obj)
                        writer_obj.writerow(data)
                        csv_obj.close()

                # else (if the csv doesn't exist):
                # make new csv, add the header row (cell + timepoints in a list), and append data
                else:
                    with open(trial_csv_name, "w+") as csv_obj:
                        writer_obj = writer(csv_obj)
                        writer_obj.writerow(header)
                        writer_obj.writerow(data)
                        csv_obj.close()


