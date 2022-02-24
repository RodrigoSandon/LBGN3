import pandas as pd
import os, glob
from typing import List, Dict
from pathlib import Path


def find_paths(root_path: Path, middle: str, endswith: str) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, "**", middle, "**", endswith), recursive=True,
    )
    return files


def binary_search(data, val):
    """Will return index if the value is found, otherwise the index of the item that is closest
    to that value."""
    lo, hi = 0, len(data) - 1
    best_ind = lo
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if data[mid] < val:
            lo = mid + 1
        elif data[mid] > val:
            hi = mid - 1
        else:
            best_ind = mid
            break
        # check if data[mid] is closer to val than data[best_ind]
        if abs(data[mid] - val) < abs(data[best_ind] - val):
            best_ind = mid

    return best_ind


def binary_search2(arr, low, high, x):

    # Check base case
    if high >= low:

        mid = (high + low) // 2

        # If element is present at the middle itself
        if arr[mid] == x:
            return mid

        # If element is smaller than mid, then it can only
        # be present in left subarray
        elif arr[mid] > x:
            return binary_search2(arr, low, mid - 1, x)

        # Else the element can only be present in right subarray
        else:
            return binary_search2(arr, mid + 1, high, x)

    else:
        # Element is not present in the array
        return -1


class EventDataset:
    def __init__(self, event_category, d):
        self.event_category = event_category
        self.d = d


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
    ):
        self.block = block
        self.event_category = event_category
        self.outcome = outcome
        self.mouse = mouse
        self.session = session
        self.trial_number = trial_number
        self.cell = cell
        self.trial_dff_trace = dff_trace

        self.prechoice_dff_trace = self.get_prechoice_dff_trace()
        self.postchoice_dff_trace = self.get_postchoice_dff_trace()

    def get_prechoice_dff_trace(self):
        """Gives you activity at beginning (-10s) to 0"""
        pass

    def get_postchoice_dff_trace(self):
        """Gives you activity at 0 to 10s"""
        pass


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


def categorize_trial_dff_traces(all_csv_paths: Path, **kwargs: tuple):

    d: Dict[str, dict]
    d = {}


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
                            [Trial]: {
                                [Cell] : [Cell df/f timewindow],
                                .
                                .
                                .
                            }
                        },
                        [Session]: {
                            [Trial]: {
                                [Cell] : [Cell df/f timewindow],
                                .
                                .
                                .
                            }
                        }
                    },
                    [Mouse]: {
                        [Session]: {
                            [Trial]: {
                                [Cell] : [Cell df/f timewindow],
                                .
                                .
                                .
                            }
                        },
                        [Session]: {
                            [Trial]: {
                                [Cell] : [Cell df/f timewindow],
                                .
                                .
                                .
                            }
                        }
                    }
                },
            }
        }
    """

    ROOT_PATH = Path(r"/media/rory/Padlock_DT/BLA_Analysis")

    files_1 = find_paths(
        ROOT_PATH, middle="Reward Size_Choice Time (s)", endswith="plot_ready.csv"
    )

    files_2 = find_paths(
        ROOT_PATH, middle="Omission_Choice Time (s)", endswith="plot_ready.csv"
    )
    all_csv_paths = files_1 + files_2

    for csv_path in all_csv_paths:
        csv_path = Path(csv_path)
        # extract bins we can get from the csv_path itself
        block, event_category, outcome, mouse, session, cell = custom_dissect_path(
            csv_path
        )
        # open the csv and loop through the df to acquire trials
        df: pd.DataFrame
        df = pd.read_csv(csv_path)
        df = df.iloc[:, 1:]

        timepoints = [
            float(i.replace("-", "")) for i in list(df.columns) if "-" in i
        ] + [float(i) for i in list(df.columns) if "-" not in i]

        for idx, i in enumerate(timepoints):
            if i > 1000000:  # the number will in fact be greater than 0. hopefully??
                timepoints[idx] = 0

        idx_at_time_zero = binary_search(timepoints, 0)
        print("idx: ", idx_at_time_zero)
        print("timepoint: ", timepoints[idx_at_time_zero])

        for i in range(len(df)):
            trial_num = i + 1
            dff_prechoice_subwindow = None
            dff_postchoice_subwindow = None
            new_trial = Trial(
                block,
                event_category,
                outcome,
                mouse,
                session,
                trial_num,
                cell,
                list(df.iloc[i, :]),
            )
            print(list(df.iloc[i, :])[0 : idx_at_time_zero + 2])


def tester():
    example = "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/Post-RDT D3/SingleCellAlignmentData/C02/Block_Reward Size_Choice Time (s)/(3.0, 'Large')/plot_ready.csv"

    csv_path = Path(example)
    # extract bins we can get from the csv_path itself
    block, event_category, outcome, mouse, session, cell = custom_dissect_path(csv_path)
    # open the csv and loop through the df to acquire trials
    df: pd.DataFrame
    df = pd.read_csv(csv_path)
    df = df.iloc[:, 1:]
    print(df.head())
    # print(list(df.iloc[0, :]))  # choosing one row (all of it)

    # choose a subset of each row based on column
    # col names to list
    # search for zero using binary search, get the index, this is the max index (inclusive)
    timepoints = [float(i.replace("-", "")) for i in list(df.columns) if "-" in i] + [
        float(i) for i in list(df.columns) if "-" not in i
    ]
    # so when I did this the zero became the greatest number bc it was E-17 something
    for idx, i in enumerate(timepoints):
        if i > 1000000:  # the number will in fact be greater than 0. hopefully??
            timepoints[idx] = 0

    # print(timepoints)
    # print(len(timepoints))
    # current timepoints are not ints
    idx_at_time_zero = binary_search(timepoints, 0)
    print("idx: ", idx_at_time_zero)
    print("timepoint: ", timepoints[idx_at_time_zero])

    print(timepoints[0 : idx_at_time_zero + 2])
    print(len(timepoints[0 : idx_at_time_zero + 2]))
    print(list(df.iloc[0, :])[0 : idx_at_time_zero + 2])

    # almost a hardcode but it's bc the binary search is not finding the lowest value


if __name__ == "__main__":
    # main()
    tester()
