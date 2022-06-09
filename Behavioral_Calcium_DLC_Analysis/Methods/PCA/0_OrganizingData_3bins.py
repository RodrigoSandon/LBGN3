from re import I
import pandas as pd
import os, glob
from typing import List, Dict
from pathlib import Path
from csv import writer, DictWriter


def find_paths(root_path: Path, middle_1, middle_2, endswith: str) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, "**", middle_1, "**",middle_2, "**", endswith), recursive=True,
    )
    return files


def find_paths_endswith(root_path, endswith) -> List:

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )

    return files


"""
The thing is, i'm supposed to feed in the single trial here.
Oh okay so I add on all the existing trials first for one cell,
and as i'm going through this one session, I append to those trials with different cells.
"""


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

        """self.prechoice_dff_trace = self.get_prechoice_dff_trace()
        self.postchoice_dff_trace = self.get_postchoice_dff_trace()

        self.prechoice_timepoints = self.get_prechoice_timepoints()
        self.postchoice_timepoints = self.get_postchoice_timepoints()"""

    def get_prechoice_dff_trace(self):
        """Gives you activity at beginning (-3s) to 0"""
        return self.trial_dff_trace[70 : self.idx_at_time_zero + 1]

    def get_postchoice_dff_trace(self):
        """Gives you activity at 0 to 10s"""
        return self.trial_dff_trace[self.idx_at_time_zero + 1 : -1]

    def get_prechoice_timepoints(self):
        return self.timepoints[70 : self.idx_at_time_zero + 1]

    def get_postchoice_timepoints(self):
        return self.timepoints[self.idx_at_time_zero + 1 : -1]


def strip_outcome(my_str):
    outcome_bins: list
    if "(" in my_str:
        my_str = my_str.replace("(", "")
    if ")" in my_str:
        my_str = my_str.replace(")", "")
    if "'" in my_str:
        my_str = my_str.replace("'", "")
    if ", " in my_str:
        outcome_bins = my_str.split(", ")

    breakdown_1 = outcome_bins[0]
    breakdown_2 = outcome_bins[1]
    breakdown_3 = outcome_bins[2]
    # print(outcome_bins)
    return breakdown_1, breakdown_2, breakdown_3


# /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-9/RDT D1/SingleCellAlignmentData/C02/Block_Reward Size_Choice Time (s)/(2.0, 'Large')/plot_ready.csv
def custom_dissect_path(path: Path):
    parts = list(path.parts)
    event_category = parts[10]
    block, outcome, shock = strip_outcome(parts[11])
    mouse = parts[6]
    session = parts[7]
    cell = parts[9]
    # Now you actually need to go into the the csv file to get the events you want.
    # Do this in another process
    return block, event_category, outcome, shock, mouse, session, cell


def main():

    ROOT_PATH = Path(r"/media/rory/Padlock_DT/BLA_Analysis")
    DST_PATH = Path(r"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Unnorm_3Bin_Arranged_Dataset_-10_10")

    sessions = ["Pre-RDT RM"]

    all_csv_paths = []

    for session in sessions:

        files = find_paths(
            ROOT_PATH, middle_1=f"{session}/SingleCellAlignmentData", middle_2="Block_Reward Size_Shock Ocurred_Choice Time (s)", endswith="plot_ready.csv"
        )

        all_csv_paths += files

    #print( all_csv_paths)
    for csv_path in all_csv_paths:
        try:
            csv_path = Path(csv_path)
            # extract bins we can get from the csv_path itself
            block, event_category, outcome, shock, mouse, session, cell = custom_dissect_path(
                csv_path
            )
            # Now create the folders where this info of these trials will go
            # including event_category in dirs is not neccesary, we will be able to make it out by
            # looking at the dir structure itself
            new_dirs = os.path.join(DST_PATH, block, outcome, shock, mouse, session)
            print(f"Dirs being created: {new_dirs}")
            os.makedirs(new_dirs, exist_ok=True)
            # open the csv and loop through the df to acquire trials
            df: pd.DataFrame
            df = pd.read_csv(csv_path)
            df = df.iloc[:, 1:]



            timepoints = [
                float(i.replace("-", "")) for i in list(df.columns) if "-" in i
            ] + [float(i) for i in list(df.columns) if "-" not in i]

            idx_at_time_zero: int
            for idx, i in enumerate(timepoints):
                # forcing numbers to be negative as we go since they were initially negative
                timepoints[idx] = -abs(i)
                if i > 1000000:  # timepoint values will not change so ur good
                    idx_at_time_zero = idx
                    # changing last number to be zero b/c it is in fact zero (not some v. high number)
                    timepoints[idx] = 0

            # setting pos values after zero
            for idx, i in enumerate(timepoints):
                if idx > idx_at_time_zero:
                    timepoints[idx] = abs(i)
                    
            #print(timepoints)
            #print("idx: ", idx_at_time_zero)
            #
            # print("timepoint: ", timepoints[idx_at_time_zero])

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

                ### CHANGE HERE TO GET SPECIFIC TIME WINDOW YOU WANT ###
                header = ["Cell"] + new_trial.timepoints[:]
                data = [cell] + new_trial.trial_dff_trace[:]
                ### CHANGE HERE TO GET SPECIFIC TIME WINDOW YOU WANT ###

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
        except IndexError as e:
            # Very rarely, the csv that's opened contains no traces at all--how could this be?
            # ex: bla6, rm d2, block_rew size, any cell, and look at the plot_ready.csv
            print(e)
            pass

if __name__ == "__main__":
    main()