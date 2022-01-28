import pandas as pd
from scipy import stats

# Initiate Cell obj with values of column
# Purpose: to compare across Cell objs based on attributes
class Cell:
    def __init__(
        self,
        cell_name,
        dff_traces: list,
        unknown_time_min,
        unknown_time_max,
        reference_pair: dict,
        hertz: int,
    ):
        self.cell_name = cell_name
        self.dff_traces = dff_traces
        # self.cell_dict = {cell_name: dff_traces}

        self.unknown_time_min = unknown_time_min
        self.unknown_time_max = unknown_time_max
        self.reference_pair = reference_pair
        self.hertz = hertz

        self.arr_of_focus = self.make_arr_of_focus()
        self.z_score = self.average_zscore()  # based on the time window

    def average_zscore(self) -> float:
        is_normal_dist = self.is_normal_distribution()
        mu = 0
        sigma = 1

        if is_normal_dist == True:
            return (stats.tmean(self.arr_of_focus) - mu) / sigma
        elif is_normal_dist == False:
            print("Not a normal distribution!")
            pass

    def is_normal_distribution(self) -> bool:
        if len(self.arr_of_focus) >= 30:
            # print(f"ARR OF FOCUS LENGTH: {len(self.arr_of_focus)}")
            if len(self.arr_of_focus) > 30:
                print("Over 30 samples!")
            return True
        else:
            print(f"{self.cell_name} not in a norm dist!")
            return True

    def make_arr_of_focus(self):
        reference_time = list(self.reference_pair.keys())[0]  # has to come from 0
        reference_idx = list(self.reference_pair.values())[0]

        idx_start = (self.unknown_time_min * self.hertz) + reference_idx
        # idx_end = (self.unknown_time_max * self.hertz) + reference_idx + 1 ? 11/30/21
        idx_end = (self.unknown_time_max * self.hertz) + reference_idx

        return self.dff_traces[int(idx_start) : int(idx_end)]
