from re import I
import pandas as pd
import os, glob
from typing import List, Dict
from pathlib import Path
from csv import writer, DictWriter


def find_paths(root_path: Path, middle: str, endswith: str) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, "**", middle, "**", endswith), recursive=True,
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
    # print(outcome_bins)
    return breakdown_1, breakdown_2


# /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-9/RDT D1/SingleCellAlignmentData/C02/Block_Reward Size_Choice Time (s)/(2.0, 'Large')/plot_ready.csv
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


# /media/rory/Padlock_DT/BLA_Analysis/Decoding/Pre-Arranged_Dataset/Shock Test/Shock/0.22-0.3/BLA-Insc-2/C03/plot_ready.csv
def shock_dissect_path(path: Path):
    parts = list(path.parts)
    session = parts[7]
    outcome = parts[8]
    intensity = parts[9]
    mouse = parts[10]
    cell = parts[11]
    return session, outcome, intensity, mouse, cell


def main():

    """
    Goal: To have the decoder predict whether the mouse will choose Large, Small, or Omit based on before-choice activity.
    Training Data: Any data that will lead to a choice of L/S/O. Therefore includes these categorizations of data:
        1) Block_Omission_Choice Time (s)
        2) Block_Reward Size_Choice Time (s)

    But if we use these categorizations, we will be essentially repeating some data b/c the more generalized subcategorizations will include subcategorization of more
    specific subcategorizations. 

    The reason we may want to now include the data that are controlling -- wait nvm. It's better if we used the most generalized dataset b/c then we are 
    inputting inputs that can be of any trial type, whether shock ocurred or not, and any block (shock probability).

    But we would also like to see how these matrices change across blocks, so we would have to encase the entire dict under
    different blocks. Would then have to  have **kwargs of values that correspond to the additional categorization type.
    You would input it as the following: block=("Block", [1,2,3]), shock_ocurred=("Shock", [True, False]), etc ...
    (btw, shock shouldn't matter b/c we'll be selecting prechoice neural activity)

    The first key should be the first breakdown, the second key should be the 2nd breakdown and so on. But no matter how many
    breakdowns, at the core, there is the outcome/state we want to predict that is broken down like this:

        outcome -> mouse -> session -> trial -> list of dff for all cells meeting the same criteria for all the prev criteria

    I will flatten this core structure down eventually where the csvs will have their according names pertaining to where they
    came from:

        outcome -> bla9_rdtd1_plot_ready.csv || outcome -> bla9_rdtd1_trial1_trialmatrix.csv
    """

    # So now, if i get into one of these csv files under the same mouse, session, outcome, and same trial, I can start inputting them into a dict
    """
    Dict will look something like this:

        d = {
            [Block]: {
                [Outcome]: {
                    [Mouse]: {
                        [Session]: {
                            [Trial 1]: {
                                [Cell] : [Cell df/f timewindow],
                                .
                                .
                                .
                            }
                            [Trial 2]: {
                                [Cell] : [Cell df/f timewindow],
                                .
                                .
                                .
                            }
                        },
                    },
                },
            }
        }
    """
    # a trial has it's own csv

    ROOT_PATH = Path(r"/media/rory/Padlock_DT/BLA_Analysis")
    DST_PATH = Path(r"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Arranged_Dataset")

    files_1 = find_paths(
        ROOT_PATH, middle="Block_Reward Size_Choice Time (s)", endswith="plot_ready.csv"
    )

    files_2 = find_paths(
        ROOT_PATH, middle="Block_Omission_Choice Time (s)", endswith="plot_ready.csv"
    )
    all_csv_paths = files_1 + files_2

    for csv_path in all_csv_paths:
        try:
            csv_path = Path(csv_path)
            # extract bins we can get from the csv_path itself
            block, event_category, outcome, mouse, session, cell = custom_dissect_path(
                csv_path
            )
            # Now create the folders where this info of these trials will go
            # including event_category in dirs is not neccesary, we will be able to make it out by
            # looking at the dir structure itself
            new_dirs = os.path.join(DST_PATH, block, outcome, mouse, session)
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

            print(timepoints)
            print("idx: ", idx_at_time_zero)
            print("timepoint: ", timepoints[idx_at_time_zero])

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
        except IndexError as e:
            # Very rarely, the csv that's opened contains no traces at all--how could this be?
            # ex: bla6, rm d2, block_rew size, any cell, and look at the plot_ready.csv
            print(e)
            pass


def main_shock():
    """MAKE SURE TO DELETE ARRANGED_DATASET FOR SHOCK TEST TO GET NEW RESULTS"""
    ROOT_PATH = Path(
        r"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Pre-Arranged_Dataset"
    )
    DST_PATH = Path(r"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Arranged_Dataset")

    all_csv_paths = find_paths_endswith(ROOT_PATH, endswith="plot_ready.csv")

    for csv_path in all_csv_paths:
        try:
            csv_path = Path(csv_path)
            # extract bins we can get from the csv_path itself
            session, outcome, intensity, mouse, cell = shock_dissect_path(csv_path)
            # Now create the folders where this info of these trials will go
            # including event_category in dirs is not neccesary, we will be able to make it out by
            # looking at the dir structure itself
            new_dirs = os.path.join(DST_PATH, session, outcome, intensity, mouse)
            print(f"Dirs being created: {new_dirs}")
            os.makedirs(new_dirs, exist_ok=True)
            # open the csv and loop through the df to acquire trials
            df: pd.DataFrame
            df = pd.read_csv(csv_path)

            df = df.iloc[:, 1:]  # remove first event # col

            # In the case of shock test session, i've already cut out the parts we want
            # in another script

            for i in range(len(df)):
                trial_num = i + 1

                trial_csv_name = os.path.join(new_dirs, f"trial_{trial_num}.csv")
                #print(list(df.columns))
                header = ["Cell"] + list(df.columns)
                # get the current row (but not the timepoints)
                data = [cell] + list(df.iloc[i, :])
                # print(len(list(df.iloc[i, :])))

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


def tester():
    example = "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/Post-RDT D3/SingleCellAlignmentData/C02/Block_Reward Size_Choice Time (s)/(3.0, 'Large')/plot_ready.csv"

    csv_path = Path(example)
    # extract bins we can get from the csv_path itself
    block, event_category, outcome, mouse, session, cell = custom_dissect_path(csv_path)
    # open the csv and loop through the df to acquire trials
    df: pd.DataFrame
    df = pd.read_csv(csv_path)
    df = df.iloc[:, 1:]
    # print(df.head())
    # print(list(df.iloc[0, :]))  # choosing one row (all of it)

    # choose a subset of each row based on column
    # col names to list
    # search for zero using binary search, get the index, this is the max index (inclusive)
    timepoints = [float(i.replace("-", "")) for i in list(df.columns) if "-" in i] + [
        float(i) for i in list(df.columns) if "-" not in i
    ]
    # so when I did this the zero became the greatest number bc it was E-17 something
    idx_at_time_zero: int
    for idx, i in enumerate(timepoints):
        if i > 1000000:
            # the number will in fact be greater than 0. hopefully?? Shouldn't change bc timepoints will always be the same
            idx_at_time_zero = idx

    # print(timepoints)
    print(len(timepoints))
    # current timepoints are not ints
    # print("idx: ", idx_at_time_zero)
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
            idx_at_time_zero,
        )

        # print(new_trial.prechoice_dff_trace)
        print(len(new_trial.prechoice_dff_trace))
        # Now a new trial obj is being made for each csv, so now start building your dict
        break


if __name__ == "__main__":
    # main()
    main_shock()
    # tester()
