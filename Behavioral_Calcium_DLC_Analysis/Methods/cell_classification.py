import pandas as pd
import numpy as np
import math
import random

from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt
from operator import attrgetter
from scipy import stats


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
            return False

    def make_arr_of_focus(self):
        reference_time = list(self.reference_pair.keys())[0]  # has to come from 0
        reference_idx = list(self.reference_pair.values())[0]

        idx_start = (self.unknown_time_min * self.hertz) + reference_idx
        # idx_end = (self.unknown_time_max * self.hertz) + reference_idx + 1 ? 11/30/21
        idx_end = (self.unknown_time_max * self.hertz) + reference_idx

        return self.dff_traces[int(idx_start) : int(idx_end)]


class Utilities:
    def change_cell_names(df):

        for col in df.columns:

            df = df.rename(columns={col: col.replace("BLA-Insc-", "")})
            # print(col)

        return df

    def zscore(obs_value, mu, sigma):
        return (obs_value - mu) / sigma

    def convert_secs_to_idx(
        unknown_time_min, unknown_time_max, reference_pair: dict, hertz: int
    ):
        reference_time = list(reference_pair.keys())[0]  # has to come from 0
        reference_idx = list(reference_pair.values())[0]

        idx_start = (unknown_time_min * hertz) + reference_idx

        idx_end = (unknown_time_max * hertz) + reference_idx  # exclusive
        return int(idx_start), int(idx_end)

    def create_subwindow_for_col(
        df, col, unknown_time_min, unknown_time_max, reference_pair, hertz
    ) -> list:
        idx_start, idx_end = Utilities.convert_secs_to_idx(
            unknown_time_min, unknown_time_max, reference_pair, hertz
        )
        subwindow = df[col][idx_start:idx_end]
        return subwindow

    def create_subwindow_of_list(
        lst, unknown_time_min, unknown_time_max, reference_pair, hertz
    ) -> list:
        idx_start, idx_end = Utilities.convert_secs_to_idx(
            unknown_time_min, unknown_time_max, reference_pair, hertz
        )

        subwindow_lst = lst[idx_start:idx_end]
        return subwindow_lst

    def zscore(obs_value, mu, sigma):
        return (obs_value - mu) / sigma

    def custom_standardize(
        df, unknown_time_min, unknown_time_max, reference_pair: dict, hertz: int
    ):
        for col in df.columns:
            subwindow = Utilities.create_subwindow_for_col(
                df, col, unknown_time_min, unknown_time_max, reference_pair, hertz
            )
            mean_for_cell = stats.tmean(subwindow)
            stdev_for_cell = stats.tstd(subwindow)

            new_col_vals = []
            for ele in list(df[col]):
                z_value = Utilities.zscore(ele, mean_for_cell, stdev_for_cell)
                new_col_vals.append(z_value)

            df[col] = new_col_vals
        return df

    def gaussian_smooth(df, sigma: float = 1.5):
        # so that it applys smoothing within a cell and not across cells
        df = df.T.apply(gaussian_filter1d, sigma=sigma, axis=0)
        # switch back to og transformation
        return df.T

    def pie_chart(
        csv_path: str, test_name: str, data: list, labels: list, replace_name: str
    ):
        fig = plt.figure(figsize=(10, 7))
        plt.pie(data, labels=labels, autopct="%1.2f%%")
        plt.title(test_name)
        new_name = csv_path.replace(".csv", replace_name)
        plt.savefig(new_name)
        plt.close()

    def make_replace_name_suffix_prefix(standardize: bool, smooth: bool):
        return f"_norm-{standardize}_smooth-{smooth}"


class CellClassification(Utilities):
    def __init__(
        self,
        csv_path: str,
        df: pd.DataFrame,
        standardize: bool,
        smooth: bool,
        lower_bound_time: int,
        upper_bound_time: int,
        reference_pair: dict,
        hertz: int,
        test: str,
    ):

        super().__init__()

        self.csv_path = csv_path
        self.standardize = standardize
        self.smooth = smooth
        # for the time window we are doing the tests based off of
        self.lower_bound_time = lower_bound_time
        # for the time window we are doing the tests based off of
        self.upper_bound_time = upper_bound_time
        # to know how to convert secs to idx, should be 0:100, 100 to account for exclusivity at end
        # so if 0:100, then it's interpreted as 0:99
        self.reference_pair = reference_pair
        self.hertz = hertz  # helps in calculating conversion

        if standardize == True and smooth == True:  # major
            self.df = Utilities.custom_standardize(
                df, lower_bound_time, upper_bound_time, reference_pair, hertz
            )
            self.df = Utilities.gaussian_smooth(self.df)
        elif standardize == False and smooth == False:  # major
            self.df = df
        elif standardize == True and smooth == False:
            self.df = Utilities.custom_standardize(
                df, lower_bound_time, upper_bound_time, reference_pair, hertz
            )
        elif standardize == False and smooth == True:
            self.df = Utilities.gaussian_smooth(df)

        if test == "stdev binary test":
            CellClassification.stdev_difference_test(self)
            CellClassification.stdev_difference_test_shuffled(self)
        elif test == "two sample t test":
            CellClassification.two_sample_t_test(self)
        elif test == "one sample t test":
            CellClassification.one_sample_t_test(self)
        elif test == "wilcoxon rank sum test":
            CellClassification.wilcoxon_rank_sum(self)
        elif test == "all":
            CellClassification.stdev_difference_test(self)
            CellClassification.stdev_difference_test_shuffled(self)
            CellClassification.two_sample_t_test(self)
            CellClassification.one_sample_t_test(self)
            CellClassification.wilcoxon_rank_sum(self)

        else:
            print("Test is not available!")

    # stdev binary test for standardize and smoothed version
    def stdev_difference_test(self):
        """
        Assumes unstandardized and unsmoothed data.
        Takes a baseline subwindow (0-10s) to extract stdev from.
        Takes the after event time (0-2s) to extract stdev from.

        Find abs difference of these two stdevs and if equal to or above 1, inactive/active.
        If below 1, neutral cell.

        Caveats:
            1) Finding significance via difference of variability of data with 1 sigma of difference being significant - not standard?

        """
        active_cells = []
        inactive_cells = []
        number_cells = len(list(self.df.columns))

        for col in list(self.df.columns):
            mean_of_entire_cell = stats.tmean(list(self.df[col]))

            sub_df_baseline_lst = Utilities.create_subwindow_of_list(
                list(self.df[col]),
                unknown_time_min=-10,
                unknown_time_max=0,
                reference_pair={0: 100},
                hertz=10,
            )
            # print(f"Shuffled data: {sub_df_rand_lst}")

            base_mean = stats.tmean(sub_df_baseline_lst)
            base_stdev = stats.tstd(sub_df_baseline_lst)

            sub_df_lst = Utilities.create_subwindow_of_list(
                list(self.df[col]),
                self.lower_bound_time,
                self.upper_bound_time,
                self.reference_pair,
                self.hertz,
            )
            # print(f"Normal data: {sub_df_lst}")

            mean = stats.tmean(sub_df_lst)
            stdev = stats.tstd(sub_df_lst)

            # print(f"baseline mean for {col}: {base_mean}")
            # print(f"mean for {col}: {mean}")

            if abs(stdev - base_stdev) >= 1:
                active_cells.append(col)
            elif abs(stdev - base_stdev) < 1:
                inactive_cells.append(col)

        d = {
            "Non-Neutral Cells": len(active_cells),
            "Neutral Cells": len(inactive_cells),
        }
        replace_name_prefix = Utilities.make_replace_name_suffix_prefix(
            self.standardize, self.smooth
        )

        Utilities.pie_chart(
            self.csv_path,
            f"Sigma Difference Post-Choice vs Baseline (n={number_cells})",
            list(d.values()),
            list(d.keys()),
            replace_name=f"{replace_name_prefix}_pie.png",
        )

    def stdev_difference_test_shuffled(self):  # stdev binary test
        """
        It assumes unstandardized and unsmoothed data.
        First randomizes dff traces of cell, then takes a subwindow (0-2s) of the shuffled dff traces and finds
        its stdev. Then it finds the stdev of the unshuffled dff traces for the same subwindow.

        Finds differences of stdev, if abs(stdev difference) is equal to or above 1, its non-neutral cell, else,
        it's a neutral cell.

        Caveat:
            1) Finding significance via difference of variability of data with 1 sigma of difference being significant - not standard?
            2) Result will vary from test to test.
        """
        active_cells = []
        inactive_cells = []
        number_cells = len(list(self.df.columns))

        for col in list(self.df.columns):
            print(col)
            rand_data_lst = [i for i in list(self.df[col])]
            count = 0
            while count != 1000:
                count += 1
                random.shuffle(rand_data_lst)  # randomize the list

            sub_df_rand_lst = Utilities.create_subwindow_of_list(
                rand_data_lst,
                self.lower_bound_time,
                self.upper_bound_time,
                self.reference_pair,
                self.hertz,
            )
            # print(f"Shuffled data: {sub_df_rand_lst}")

            rand_mean = stats.tmean(sub_df_rand_lst)
            rand_stdev = stats.tstd(sub_df_rand_lst)

            sub_df_lst = Utilities.create_subwindow_of_list(
                list(self.df[col]),
                self.lower_bound_time,
                self.upper_bound_time,
                self.reference_pair,
                self.hertz,
            )
            # print(f"Normal data: {sub_df_lst}")

            mean = stats.tmean(sub_df_lst)
            stdev = stats.tstd(sub_df_lst)

            # print(f"normal mean: {mean}")
            # print(f"rand mean: {rand_mean}")

            # stdev = math.sqrt((mean - rand_mean)**2/number_cells)
            # take amean of shuffled
            # multiply

            if abs(stdev - rand_stdev) >= 1:
                active_cells.append(col)
            elif abs(stdev - rand_stdev) < 1:
                inactive_cells.append(col)

        d = {
            "Non-Neutral Cells": len(active_cells),
            "Neutral Cells": len(inactive_cells),
        }

        replace_name_prefix = Utilities.make_replace_name_suffix_prefix(
            self.standardize, self.smooth
        )

        Utilities.pie_chart(
            self.csv_path,
            f"Sigma Difference Shuffled vs Unshuffled (n={number_cells})",
            list(d.values()),
            list(d.keys()),
            replace_name=f"{replace_name_prefix}_pie_1000shuffled.png",
        )

    def two_sample_t_test(self):  # two sample t test
        """
        Parametric technique --> data has a distribution
        Under the hypothesis that two samples should be sig. equal in mean, but H1 being they are not equal.
        Takes a baseline subwindow (0-10s) to extract mean from.
        Takes the after event time (0-3s) as input.

        Could also compared shuffled vs unshuffled for the same time window.
        """
        active_cells = []
        inactive_cells = []
        number_cells = len(list(self.df.columns))

        for col in list(self.df.columns):
            mean_of_entire_cell = stats.tmean(list(self.df[col]))

            sub_df_baseline_lst = Utilities.create_subwindow_of_list(
                list(self.df[col]),
                unknown_time_min=-10,
                unknown_time_max=0,
                reference_pair={0: 100},
                hertz=10,
            )

            base_mean = stats.tmean(sub_df_baseline_lst)
            base_stdev = stats.tstd(sub_df_baseline_lst)

            sub_df_lst = Utilities.create_subwindow_of_list(
                list(self.df[col]),
                unknown_time_min=0,
                unknown_time_max=3,
                reference_pair={0: 100},
                hertz=10,
            )

            result = stats.ttest_1samp(sub_df_lst, base_mean, alternative="two-sided")

            if result.pvalue < (0.01 / number_cells):  # 0.005 * 2 = 0.01
                active_cells.append(col)
            elif result.pvalue >= (0.01 / number_cells):
                inactive_cells.append(col)

        d = {
            "Non-Neutral Cells": len(active_cells),
            "Neutral Cells": len(inactive_cells),
        }

        replace_name_prefix = Utilities.make_replace_name_suffix_prefix(
            self.standardize, self.smooth
        )

        Utilities.pie_chart(
            self.csv_path,
            f"Two-Sample T-test (n={number_cells})",
            list(d.values()),
            list(d.keys()),
            replace_name=f"{replace_name_prefix}_pie_bonferroni_two-sample_ttest.png",
        )

    def one_sample_t_test(self):  # one sample t test
        """
        Parametric technique --> data assumes a distribution
        Under the hypothesis that two samples should be sig. equal in mean, but H1 being 0-3s is greater/less.
        Takes a baseline subwindow (0-10s) to extract mean from.
        Takes the after event time (0-3s) as input.

        Could also compared shuffled vs unshuffled for the same time window.
        """

        active_cells = []
        inactive_cells = []
        neutral_cells = []
        number_cells = len(list(self.df.columns))

        for col in list(self.df.columns):
            mean_of_entire_cell = stats.tmean(list(self.df[col]))

            sub_df_baseline_lst = Utilities.create_subwindow_of_list(
                list(self.df[col]),
                unknown_time_min=-10,
                unknown_time_max=0,
                reference_pair={0: 100},
                hertz=10,
            )

            base_mean = stats.tmean(sub_df_baseline_lst)
            base_stdev = stats.tstd(sub_df_baseline_lst)

            sub_df_lst = Utilities.create_subwindow_of_list(
                list(self.df[col]),
                unknown_time_min=0,
                unknown_time_max=3,
                reference_pair={0: 100},
                hertz=10,
            )

            result_greater = stats.ttest_1samp(
                sub_df_lst, base_mean, alternative="greater"
            )

            result_less = stats.ttest_1samp(sub_df_lst, base_mean, alternative="less")

            if result_greater.pvalue < (0.01 / number_cells):  # 0.005 * 2 = 0.01
                active_cells.append(col)
            elif result_less.pvalue < (0.01 / number_cells):
                inactive_cells.append(col)
            else:
                neutral_cells.append(col)

        d = {
            "(+) Active Cells": len(active_cells),
            "(-) Active Cells": len(inactive_cells),
            "Neutral Cells": len(neutral_cells),
        }

        replace_name_prefix = Utilities.make_replace_name_suffix_prefix(
            self.standardize, self.smooth
        )

        Utilities.pie_chart(
            self.csv_path,
            f"One-Sample T-test (n={number_cells})",
            list(d.values()),
            list(d.keys()),
            replace_name=f"{replace_name_prefix}_pie_bonferroni_one-sample_ttest.png",
        )

    def wilcoxon_rank_sum(self):  # wilcoxon rank sum test
        """
        Nonparametric technique --> data does not assume a distribution
        Under the hypothesis that two samples should be sig. equal in mean, but H1 being 0-3s is greater/less.
        Takes a baseline subwindow (-10 to 0s) to extract mean from.
        Takes the after event time (0-2s) as input.

        - More robust model
        - Ranks data points and takes probability that the sum of ranks (observed distribution) for a group is less/greater than that of the
        Wilcoxon disttribution if the H1 is greater/less than Wilcoxon distribution.

        Could also compared shuffled vs unshuffled for the same time window.
        """
        active_cells = []
        inactive_cells = []
        neutral_cells = []
        number_cells = len(list(self.df.columns))

        for col in list(self.df.columns):  # a col is a cell

            sub_df_baseline_lst = Utilities.create_subwindow_of_list(
                list(self.df[col]),
                unknown_time_min=-10,
                unknown_time_max=0,
                reference_pair={0: 100},
                hertz=10,
            )

            sub_df_lst = Utilities.create_subwindow_of_list(
                list(self.df[col]),
                unknown_time_min=0,
                unknown_time_max=2,
                reference_pair={0: 100},
                hertz=10,
            )

            result_greater = stats.mannwhitneyu(
                sub_df_lst, sub_df_baseline_lst, alternative="greater"
            )

            result_less = stats.mannwhitneyu(
                sub_df_lst, sub_df_baseline_lst, alternative="less"
            )

            if result_greater.pvalue < (0.01 / number_cells):  # 0.005 * 2 = 0.01
                active_cells.append(col)
            elif result_less.pvalue < (0.01 / number_cells):
                inactive_cells.append(col)
            else:
                neutral_cells.append(col)

        d = {
            "(+) Active Cells": len(active_cells),
            "(-) Active Cells": len(inactive_cells),
            "Neutral Cells": len(neutral_cells),
        }

        replace_name_prefix = Utilities.make_replace_name_suffix_prefix(
            self.standardize, self.smooth
        )

        Utilities.pie_chart(
            self.csv_path,
            f"Wilcoxon Rank Sum Test (n={number_cells})",
            list(d.values()),
            list(d.keys()),
            replace_name=f"{replace_name_prefix}_0-2s_-10-0s_bonf_pie_manwhitney_010722.png",
        )


def main():
    """
    Notes:
        F-test
            1) F test was cosidered to compare variance (similar to variability via stdev), we can say the
                population is norm distributed: N >= 30 --> this can be met
            2) Samples need to be taken at random (one sample is, but another is sequentially chosen)
            3) The samples are independent (random sample not affected by a sequential one)
            http://www2.psychology.uiowa.edu/faculty/mordkoff/GradStats/part%201/I.07%20normal.pdf

    Facts:
        SD vs Variance
         1. The SD is usually more useful to describe the variability of the data while the variance is usually much more useful mathematically.

    """

    CONCAT_CELLS_PATH = r"/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/RDT D2/Shock Ocurred_Choice Time (s)/True/all_concat_cells.csv"
    df = pd.read_csv(CONCAT_CELLS_PATH)
    df = Utilities.change_cell_names(df)

    CellClassification(
        CONCAT_CELLS_PATH,
        df,
        standardize=False,
        smooth=False,
        lower_bound_time=0,
        upper_bound_time=2,
        reference_pair={0: 100},
        hertz=10,
        test="wilcoxon rank sum test",
    )


if __name__ == "__main__":
    main()
