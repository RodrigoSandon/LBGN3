from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np
import time
import random
import glob
import os

from scipy.ndimage.filters import gaussian_filter1d

from scipy.stats.stats import tmean


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


def find_paths_startswith(root_path, startswith) -> List:

    files = glob.glob(
        os.path.join(root_path, "**", "%s*") % (startswith),
        recursive=True,
    )

    return files


def extract_certain_paths(paths_list: list, **kwargs) -> list:
    kept_paths = []

    must_contain = []
    for groupby_key, value in kwargs.items():
        must_contain.append(value)

    for path in paths_list:
        if all(sub_str in path for sub_str in must_contain):
            # indicates that this path must contain all the parameters specified
            print(f"Identified: {path}")
            kept_paths.append(path)

    return kept_paths


def avg_df_cols(
    df: pd.DataFrame, include_first_col: bool, iter_num: Optional[int] = None
) -> list:
    if include_first_col == False:
        df = df.iloc[:, 1:]  # to exclude the first column, which is event #

    avg_of_cols = []
    # Iterates through columns, finding means, and appending to an array
    for col_name, col_data in df.iteritems():
        avg_dff_of_timewindow_of_event = df[col_name].mean()
        avg_of_cols.append(avg_dff_of_timewindow_of_event)

    # print(f"Average for iter {iter_num}: {avg_of_cols}")
    return avg_of_cols


def plot_trace(csv_path: str, out_path: str):
    df = pd.read_csv(csv_path)

    xaxis = list(df.index.tolist())
    plt.plot(xaxis, list(df[df.columns[0]]))
    plt.title(("Average DF/F Trace for %s's Event Window") % (df.columns[0]))
    plt.xlabel("Time (s)")
    plt.ylabel("Average DF/F of All Events")
    plt.savefig(out_path)
    plt.close()


def coeff_variation(lst_of_nums: list):
    return stats.variation(lst_of_nums)


def acquire_avg_shuffled_eventraces_for_cell(
    csv_path: str, shuffle_iters: int, new_csv_name: str
):

    print(f"Working on: {csv_path}")
    start = time.time()
    # start_1 = time.time()
    df_plot_ready = pd.read_csv(csv_path)
    # end_2 = time.time()
    # print(f"Time to load_csv: {end_2 - start_1}")

    df_avg_of_shuffled_iterations = {}
    num_iters = 0

    while num_iters < shuffle_iters:
        num_iters += 1
        # print(num_iters)

        # a copy of df is needed
        deep_copy = df_plot_ready.copy(deep=True)
        deep_copy = deep_copy.T
        # Now events are cols and rows are time point

        for event in list(deep_copy.columns):  # goes through all columns
            random.shuffle(
                deep_copy[event][1:]
            )  # <- make sure you are not including the Event name in the mix of shuffling
            # Avoid this by not including the first row into the shuffle
        deep_copy = deep_copy.T
        # take the average of all these shuffled events
        avg_of_events_for_iter = avg_df_cols(deep_copy, include_first_col=False)
        df_avg_of_shuffled_iterations[f"Iter {num_iters}"] = avg_of_events_for_iter

    # Now have big 1000 X 200, average this out
    big_df = pd.DataFrame.from_dict(df_avg_of_shuffled_iterations)
    big_df = big_df.T
    # print(big_df.head())
    # print(big_df.columns)
    avg_iters = avg_df_cols(big_df, include_first_col=True)
    # calculate coefficeint of variation here
    coeff_var = coeff_variation(avg_iters)
    print(f"Coefficient of variation is {coeff_var} for {shuffle_iters} iterations.")
    # Save the df
    # /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-7/Pre-RDT RM/SingleCellAlignmentData/C01/Block_Choice Time (s)/1.0/plot_ready.csv
    cell_name = csv_path.split("/")[9]
    df = pd.DataFrame(avg_iters, columns=[cell_name])
    out_path = csv_path.replace("plot_ready.csv", new_csv_name)
    df.to_csv(out_path, index=False)

    end = time.time()
    print(f"Time taken for {csv_path}: {(end - start)} s")


def create_concat_of_certain_shuf_cells(csvs: list, max_iters: int, dest_path: str):
    # /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-3/RDT D2/SingleCellAlignmentData/C01/Shock Ocurred_Choice Time (s)/True/shuf1000_avg_plot_ready.csv
    concat_cells = {}

    for csv in csvs:
        mouse = csv.split("/")[6]
        cell_name = csv.split("/")[9]
        col_name = mouse + "_" + cell_name

        df = pd.read_csv(csv)

        concat_cells[col_name] = df[cell_name].tolist()

    all_cells_df = pd.DataFrame.from_dict(concat_cells)
    out_path = os.path.join(dest_path, f"all_concat_cells_shuf{max_iters}.csv")
    all_cells_df.to_csv(out_path, index=False)


def shuf_all_concat_cells_comparison(
    og_concat_path: str,
    shuf_concat_path: str,
    SD_difference: int,
    max_iters: int,
    plot_out_path: str,
):
    og_df = pd.read_csv(og_concat_path)
    shuf_df = pd.read_csv(shuf_concat_path)

    posActive = []
    negActive = []
    neutral = []

    num_cells = 0
    for col in list(
        shuf_df.columns
    ):  # both dfs should have the same col names, ordered the same
        num_cells += 1
        shuf_mean = stats.tmean(shuf_df[col].tolist())
        shuf_sd = stats.tstd(shuf_df[col].tolist())

        upperSD = shuf_mean + (SD_difference * shuf_sd)
        lowerSD = shuf_mean - (SD_difference * shuf_sd)

        sub_df_og = Utilities.create_subwindow_of_list(
            og_df[col].tolist(),
            unknown_time_min=0,
            unknown_time_max=2,
            reference_pair={0: 100},
            hertz=10,
        )  # now aquiring unshuffled all concat cells

        og_mean = stats.tmean(
            sub_df_og
        )  # <- need to open unshuffled all_concat_cells.csv for this

        if og_mean > upperSD:
            posActive.append(col)
        elif og_mean < lowerSD:
            negActive.append(col)
        else:
            neutral.append(col)
    d = {
        "+ Active Cells": len(posActive),
        "- Active Cells": len(negActive),
        "Neutral Cells": len(neutral),
    }

    # saves pie plot to a given path
    fig = plt.figure(figsize=(10, 7))
    plt.pie(list(d.values()), labels=list(d.keys()), autopct="%1.2f%%")
    plt.title(f"Shuffled {max_iters} vs Unshuffled (n={num_cells})")
    plt.savefig(plot_out_path)
    plt.close()


def description():
    string = """
    Functionalities:
    1) Returns list of all plot ready csv file paths, example:

    +---------+----------+----------+----------+----------+
    | Event # | -10      | -9.9     | -9.8     | -9.7     |...
    +---------+----------+----------+----------+----------+
    | Event 1 | 1.25548  | 2.172035 | 3.490741 | 3.742469 |...
    +---------+----------+----------+----------+----------+
    | Event 2 | 2.234204 | 3.117542 | 2.026499 | 4.696948 |...
    +---------+----------+----------+----------+----------+
    | Event 3 | 4.770984 | 4.456681 | 5.656993 | 4.127113 |...
    +---------+----------+----------+----------+----------+

    For every mouse:
        For every session type:
            For every cell:
                For every event category:
                    For every event type:
                        df_avg_of_shuffled_event_iterations = {
                            "iter 1" = [...],
                            .
                            .
                            .
                        }
                        # will look like this:
                            +---------+----------+----------+----------+----------+
                            | Iteration # | -10      | -9.9     | -9.8     | -9.7     |...
                            +---------+----------+----------+----------+----------+
                            | 1           | 1.25548  | 2.172035 | 3.490741 | 3.742469 |...
                            +---------+----------+----------+----------+----------+
                            .
                            .
                            .
                            +---------+----------+----------+----------+----------+
                            | 1000        | 4.770984 | 4.456681 | 5.656993 | 4.127113 |...
                            +---------+----------+----------+----------+----------+
                        For every row (event) in plot_ready.csv:
                            Shuffle the row of this event

                        avg_for_iter = []
                        avg_for_iter.append(avg of all rows)

                        #now have 1*200 list, because all the events have been averaged to time point (200 time points)

                    # Now have big 1000 X 200 table of averages for iterations

    Run BetweenCellAlignment.py to get the concatenated traces of cells for each event type (will look for
    the avg shuffled traces of each cell for every event (avg_plot_ready.csv:

        +----------------+
        | C01            |
        +----------------+
        | 6.0267928625   |
        +----------------+
        | 5.748872253125 |
        +----------------+
        | 6.10643181875  |
        +----------------+
        | 5.942177175    |
        +----------------+
                .
                .
                .

    ))

    Once have concatenated cells from running BetweenCellAlignment.py like so:

    +------------------+-------------------+-------------------+
    | BLA-Insc-2_C01   | BLA-Insc-2_C02    | BLA-Insc-5_C01    |...
    +------------------+-------------------+-------------------+
    | 1.5908035972973  | 0.903808762162162 | 0.395520983       |...
    +------------------+-------------------+-------------------+
    | 1.85934741910811 | 1.16783503918919  | 0.554314586666667 |...
    +------------------+-------------------+-------------------+
    | 1.67148849513513 | 1.11240769783784  | 0.676555127666666 |...
    +------------------+-------------------+-------------------+
    | 1.74423583540541 | 0.962307342432432 | 0.814232835666667 |...
    +------------------+-------------------+-------------------+

    For every session type:
        For every event category:
            For every event type:
                posActive = []
                negActive = []
                neutral = []
                For every column (avg cell trace):
                    upperSD = mean of shuffled dff list (entire window) + (how many stdevs u care about * stdev)
                    lowerSD = mean of shuffled dff list (entire window) - (how many stdevs u care about * stdev)
                    # so ig 1 stdev is more of a unit of the mean for that pop rather than a unit on its own?

                    empirical_response = mean of subwindow (unshuffled) <- need to open unshuffled all_concat_cells.csv for this
                    if empirical_response > upperSD:
                        posActive.append(column)
                    elif empirical_response < lowerSD:
                        negActive.append(column)
                    else:
                        neutral.append(column)
                d = {
                    "+ Active Cells": len(posActive),
                    "- Active Cells": len(negActive),
                    "Neutral Cells": len(neutral)
                }

                #to help name the path for the output of this program
                replace_name_prefix = Utilities.make_replace_name_suffix_prefix(
                    self.standardize, self.smooth)

                #saves pie plot to a given path
                Utilities.pie_chart(self.csv_path, f"Sigma Difference Shuffled vs Unshuffled (n={number_cells})", list(
                    d.values()), list(d.keys()), replace_name=f"{replace_name_prefix}_pie_1000shuffled.png")

    """

    return string


def do_everything():
    # /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/RDT D2 NEW_SCOPE/SingleCellAlignmentData/C01/Shock Ocurred_Choice Time (s)/True/plot_ready.csv

    MASTER_ROOT = r"/media/rory/Padlock_DT/BLA_Analysis/"
    all_plot_ready_csvs = find_paths_startswith(MASTER_ROOT, "plot_ready.csv")
    max_iters = 1000

    for path in all_plot_ready_csvs:
        acquire_avg_shuffled_eventraces_for_cell(
            path,
            shuffle_iters=max_iters,
            new_csv_name=f"shuf{max_iters}_avg_plot_ready.csv",
        )


def do_one_csv():
    csv_path = r"/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/Late Shock D1/SingleCellAlignmentData/C01/Block_Choice Time (s)/1.0/plot_ready.csv"
    max_iters = 1000
    name_of_file = csv_path.split("/")[-1]
    new_name = f"shuf{max_iters}_avg_plot_ready.csv"
    new_path = csv_path.replace(name_of_file, new_name)

    acquire_avg_shuffled_eventraces_for_cell(
        csv_path,
        shuffle_iters=max_iters,
        new_csv_name=f"shuf{max_iters}_avg_plot_ready.csv",
    )
    plot_trace(new_path, out_path=new_path.replace(".csv", ".png"))


def do_certain_csvs():

    MASTER_ROOT = r"/media/rory/Padlock_DT/BLA_Analysis/"
    all_plot_ready_csvs = find_paths_startswith(MASTER_ROOT, "plot_ready.csv")
    certain_paths = extract_certain_paths(
        all_plot_ready_csvs,
        session_type="RDT D2",
        event="Shock Ocurred_Choice Time (s)/True",
    )

    max_iters = 1000

    for csv_path in certain_paths:
        acquire_avg_shuffled_eventraces_for_cell(
            csv_path,
            shuffle_iters=max_iters,
            new_csv_name=f"shuf{max_iters}_avg_plot_ready.csv",
        )


def create_concat_cells():
    MASTER_ROOT = r"/media/rory/Padlock_DT/BLA_Analysis/"
    max_iters = 1000
    csvs = find_paths_startswith(MASTER_ROOT, f"shuf{max_iters}_avg_plot_ready.csv")
    # only performing this on certain cells that match under certain parameters
    certain_paths = extract_certain_paths(
        csvs,
        session_type="RDT D2",
        event="Shock Ocurred_Choice Time (s)/True",
    )

    destination_path = r"/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/RDT D2/Shock Ocurred_Choice Time (s)/True"
    create_concat_of_certain_shuf_cells(certain_paths, max_iters, destination_path)


def shuffle_comparison():
    csv_path = r"/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/Late Shock D1/SingleCellAlignmentData/C01/Block_Choice Time (s)/1.0/plot_ready.csv"
    tests = 0

    while tests < 5:
        tests += 1
        print(f"TEST {tests}")

        acquire_avg_shuffled_eventraces_for_cell(
            csv_path,
            shuffle_iters=1000,
            new_csv_name=f"shuf{1000}_avg_plot_ready.csv",
        )
        acquire_avg_shuffled_eventraces_for_cell(
            csv_path,
            shuffle_iters=100,
            new_csv_name=f"shuf{100}_avg_plot_ready.csv",
        )

        acquire_avg_shuffled_eventraces_for_cell(
            csv_path,
            shuffle_iters=10,
            new_csv_name=f"shuf{10}_avg_plot_ready.csv",
        )


def analyze_event_type():
    DIR = r"/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/RDT D2/Shock Ocurred_Choice Time (s)/True"
    max_iters = 1000
    SD_difference = 1

    og_concat_cells = os.path.join(DIR, "all_concat_cells.csv")
    shuf_concat_cells = os.path.join(DIR, f"all_concat_cells_shuf{max_iters}.csv")
    plt_path = f"/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/RDT D2/Shock Ocurred_Choice Time (s)/True/pie_shuf{max_iters}vs_unshuf.png"

    shuf_all_concat_cells_comparison(
        og_concat_cells, shuf_concat_cells, SD_difference, max_iters, plt_path
    )


# do_everything()
# do_one_csv()
# shuffle_comparison()
create_concat_cells()
analyze_event_type()
