import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from typing import List, Optional
from numpy.core.fromnumeric import mean
import pandas as pd
import seaborn as sns
from scipy import stats
import Cell
from operator import attrgetter
from pathlib import Path


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


def find_paths_startswith(root_path, startswith) -> List:

    files = glob.glob(
        os.path.join(root_path, "**", "%s*") % (startswith), recursive=True,
    )

    return files


def find_paths_conditional_endswith(
    root_path, og_lookfor: str, cond_lookfor: str
) -> List:

    all_files = []

    for root, dirs, files in os.walk(root_path):

        if cond_lookfor in files:
            # acquire the trunc file
            file_path = os.path.join(root, cond_lookfor)
            # print(file_path)
            all_files.append(file_path)
        elif cond_lookfor not in files:
            # acquire the og lookfor
            file_path = os.path.join(root, og_lookfor)
            all_files.append(file_path)

    return all_files


# For an individual cell
def indv_events_spaghetti_plot(lst_of_indv_event_traces_of_cell):
    for csv_path in lst_of_indv_event_traces_of_cell:
        print(csv_path)
        try:
            new_path = csv_path.replace("plot_ready.csv", "spaghetti_plot.png")
            df = pd.read_csv(csv_path)
            number_of_events = df.shape[0]
            # print("df # rows: ", len(df))
            df_without_eventcol = df.loc[:, df.columns != "Event #"]
            # print(df_without_eventcol.head())
            just_event_col = df.loc[:, df.columns == "Event #"]
            # print(just_event_col.head())
            df_no_eventcol_mod = custom_standardize(df_without_eventcol)
            df_no_eventcol_mod = gaussian_smooth(df_without_eventcol)
            # print(df_no_eventcol_mod.head())

            df = pd.concat([just_event_col, df_no_eventcol_mod], axis=1)
            df = df.T

            new_header = df.iloc[0]  # first row
            df = df[1:]  # don't include first row in new df
            df.columns = new_header
            # print(df.head())

            x = list(df.index)
            # print(x)

            # print(list(df.columns))
            for col in df.columns:
                print("col: ", col)
                if col != "Event #":
                    plt.plot(x, list(df[col]), label=col)

            plt.title("All Events for Cell (n=%s)" % (number_of_events))
            plt.locator_params(axis="x", nbins=20)
            plt.savefig(new_path)
            plt.close()

        except ValueError as e:
            print("VALUE ERROR:", e)
            pass


def sort_cells(
    df, unknown_time_min, unknown_time_max, reference_pair: dict, hertz: int
):

    # sorted_cells = {}
    sorted_cells = []

    for col in df.columns:
        cell = Cell.Cell(
            col,
            list(df[col]),
            unknown_time_min,
            unknown_time_max,
            reference_pair,
            hertz,
        )
        # sorted_cells[cell.cell_name] = cell
        sorted_cells.append(cell)

    # SORT THE LIST of CELL OBJECTS BASE ON ITS Z_SCORE ATTRIBUTE
    sorted_cells.sort(key=attrgetter("z_score"), reverse=True)

    # ORDERED CELL OBJECTS, NOW TO DATA TYPE
    # its list of cell objs
    def convert_lst_to_d(lst):
        res_dct = {}
        for count, i in enumerate(lst):
            # print("CURRENT CELL:", i.cell_name)
            # print("CURRENT DFF TRACE BEING ADDED:", i.dff_traces[0:5])
            # print(f"CURRENT {i.cell_name} Z score:", i.z_score)
            res_dct[i.cell_name] = i.dff_traces

        #print(f"NUMBER OF CELLS: {len(lst)}")
        return res_dct

    sorted_cells_d = convert_lst_to_d(sorted_cells)

    df_mod = pd.DataFrame.from_dict(
        sorted_cells_d
    )  # from_records automatically sorts by key smh
    # print(df_mod)
    return df_mod


def heatmap(
    df,
    file_path,
    out_path,
    cols_to_plot: Optional[List[str]] = None,
    cmap: str = "coolwarm",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    **heatmap_kwargs,
):

    try:
        if cols_to_plot is not None:
            df = df[cols_to_plot]

        ax = sns.heatmap(
            df.transpose(), vmin=vmin, vmax=vmax, cmap=cmap, **heatmap_kwargs
        )
        ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=8)
        ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=5)
        ax.tick_params(left=True, bottom=True)

        ax.set_ylabel("Neuron #")
        ax.set_xlabel("Time relative to choice (s)")

        plt.title(
            f"Smoothed Z-Scores of Neural Ca2+ Traces (n={len(list(df.columns))})"
        )
        plt.savefig(out_path)
        plt.close()

    except ValueError as e:

        print("VALUE ERROR:", e)
        print(f"VALUE ERROR FOR {file_path} --> MAKING HEATMAP")
        pass


# For averaged dff trace of cell, across cells
def spaghetti_plot(df, file_path, out_path):
    try:
        x = list(df.index)
        for cell in df.columns:
            # print("cell: ", cell)
            plt.plot(x, list(df[cell]), label=cell)
        number_cells = len(df.T)
        plt.title("Smoothed Z-Scores of Neural Ca2+ Traces (n=%s)" %
                  (number_cells))
        plt.xlabel("Time (s)")
        plt.ylabel("Z-Score")
        plt.locator_params(axis="x", nbins=20)
        plt.savefig(out_path)
        plt.close()

    except ValueError as e:

        print("VALUE ERROR:", e)
        print(f"VALUE ERROR FOR {file_path} --> MAKING SPAGHETTI")
        pass


"""
Example:
    reference_pair -> 0 seconds : 99 idx (reference is something we know already)
    hertz -> 10 Hz (10 cycles(recordings) / 1 sec)
"""


def custom_standardize(
    df: pd.DataFrame, unknown_time_min, unknown_time_max, reference_pair: dict, hertz: int
):
    # print(df.head())
    for col in df.columns:
        subwindow = create_subwindow_for_col(
            df, col, unknown_time_min, unknown_time_max, reference_pair, hertz
        )
        mean_for_cell = stats.tmean(subwindow)
        stdev_for_cell = stats.tstd(subwindow)
        # print(subwindow)
        # print(f"Mean {mean_for_cell} for cell {col}")
        # print(stdev_for_cell)

        new_col_vals = []
        for ele in list(df[col]):
            z_value = zscore(ele, mean_for_cell, stdev_for_cell)
            new_col_vals.append(z_value)

        # print(new_col_vals[0:10])  # has nan values
        df[col] = new_col_vals  # <- not neccesary bc of the .apply function?
    return df


def custom_standardize_limit(
    df: pd.DataFrame, unknown_time_min, unknown_time_max, reference_pair: dict, hertz: int, limit
):
    """A limit indicates when to stop z-scoring based off of the baseline."""
    limit_idx = convert_secs_to_idx_single_timepoint(
        limit, reference_pair, hertz) + 1
    for col in df.columns:
        subwindow = create_subwindow_for_col(
            df, col, unknown_time_min, unknown_time_max, reference_pair, hertz
        )
        mean_for_cell = stats.tmean(subwindow)
        stdev_for_cell = stats.tstd(subwindow)

        new_col_vals = []
        for count, ele in enumerate(list(df[col])):
            if count <= limit_idx:
                z_value = zscore(ele, mean_for_cell, stdev_for_cell)
            else:  # if outside limits of zscoring, don't zscore
                z_value = ele
            new_col_vals.append(z_value)

        df[col] = new_col_vals
    return df


def custom_standardize_limit_fixed(
        df: pd.DataFrame, baseline_min, baseline_max, limit_idx):
    """A limit indicates when to stop z-scoring based off of the baseline."""
    for col in df.columns:
        subwindow = list(df[col])[baseline_min: baseline_max + 1]

        mean_for_cell = stats.tmean(subwindow)
        stdev_for_cell = stats.tstd(subwindow)

        new_col_vals = []
        for count, ele in enumerate(list(df[col])):
            if count >= baseline_min and count <= limit_idx:
                z_value = zscore(ele, mean_for_cell, stdev_for_cell)
            else:  # if outside limits of zscoring, don't zscore
                z_value = ele
            new_col_vals.append(z_value)

        df[col] = new_col_vals
    return df


def zscore(obs_value, mu, sigma):
    return (obs_value - mu) / sigma


def convert_secs_to_idx(
    unknown_time_min, unknown_time_max, reference_pair: dict, hertz: int
):
    reference_time = list(reference_pair.keys())[0]  # has to come from 0
    reference_idx = list(reference_pair.values())[0]

    # first find the time difference between reference and unknown
    # Note: reference will
    idx_start = (unknown_time_min * hertz) + reference_idx
    # idx_end = (unknown_time_max * hertz) + reference_idx + 1
    # ^plus 1 bc getting sublist is exclusive? 11/30/21
    idx_end = (unknown_time_max * hertz) + reference_idx
    return int(idx_start), int(idx_end)


def convert_secs_to_idx_single_timepoint(
    unknown_time, reference_pair: dict, hertz: int
):
    reference_idx = list(reference_pair.values())[0]

    return (unknown_time * hertz) + reference_idx


"""def standardize(df):
    # from scipy.stats import zscore
    # ^ not specific enough for our use case

    return df.apply(zscore)"""


def gaussian_smooth(df, sigma: float = 1.5):
    from scipy.ndimage import gaussian_filter1d

    return df.apply(gaussian_filter1d, sigma=sigma, axis=0)


def change_cell_names(df):

    for col in df.columns:

        df = df.rename(columns={col: col.replace("BLA-Insc-", "")})
        # print(col)

    return df


def scatter_plot(subdf, out_path):
    # Note: the time column has been made by now for this df
    number_cells = len(list(subdf.columns))

    # Skip time column
    for col in subdf.columns:
        # print(subdf[col].tolist())
        plt.scatter(x=subdf.index.values.tolist(), y=subdf[col].tolist())

    plt.title(f"Smoothed Z-Scores of Neural Ca2+ Traces (n={number_cells})")
    plt.locator_params(axis="x", nbins=20)
    plt.savefig(out_path)
    plt.close()


def subdf_of_df(
    df, unknown_time_min, unknown_time_max, reference_pair, hertz
) -> pd.DataFrame:
    """
    d = {
        cell_name : [subwindow of dff traces]
    }
    """
    # note time index has been created?
    subdf_d = {}
    for col in df.columns:
        # print("SUBWINDOW")
        subdf_d[col] = create_subwindow_for_col(
            df, col, unknown_time_min, unknown_time_max, reference_pair, hertz
        )
        # print(subdf_d[col])
    subdf = pd.DataFrame.from_dict(subdf_d)
    # print("here")
    # print(subdf.head())
    return subdf


def create_subwindow_for_col(
    df, col, unknown_time_min, unknown_time_max, reference_pair, hertz
) -> list:
    idx_start, idx_end = convert_secs_to_idx(
        unknown_time_min, unknown_time_max, reference_pair, hertz
    )
    # print(idx_start, idx_end)
    subwindow = df[col][idx_start:idx_end]
    # print(subwindow)
    return subwindow


def insert_time_index_to_df(df, range_min, range_max, step) -> pd.DataFrame:
    x_axis = np.arange(range_min, range_max, step).tolist()
    # end shoudl be 10.1 and not 10 bc upper limit is exclusive

    middle_idx = int(len(x_axis) / 2)

    end_idx = len(x_axis) - 1
    # print(x_axis[end_idx])

    #x_axis[middle_idx] = 0
    x_axis = [round(i, 1) for i in x_axis]

    df.insert(0, "Time (s)", x_axis)
    df = df.set_index("Time (s)")

    return df


def main():

    ROOT_PATH = r"/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData"
    # ROOT_PATH = r"/Users/rodrigosandon/Documents/GitHub/LBGN/SampleData/truncating_bug"

    to_look_for_originally = "all_concat_cells.csv"
    # would only look for this is the file causing the conditional statement didn't exist
    to_look_for_conditional = "all_concat_cells_truncated.csv"

    csv_list = find_paths_conditional_endswith(
        ROOT_PATH, to_look_for_originally, to_look_for_conditional
    )
    # print(csv_list)
    # csv_list.reverse()
    for count, csv_path in enumerate(csv_list):

        print(f"Working on file {count}: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
            # df = truncate_past_len_threshold(df, len_threshold=200)

            df = change_cell_names(df)

            df = custom_standardize(
                df,
                unknown_time_min=-10.0,
                unknown_time_max=0.0,
                reference_pair={0: 100},
                hertz=10,
            )  # changed unknown time max: -1 to 0 3/22/22

            df = gaussian_smooth(df.T)
            df = df.T
            # print(df.head())
            # We're essentially gettin the mean of z-score for a time frame to sort
            df_sorted = sort_cells(
                df,
                unknown_time_min=0.0,
                unknown_time_max=3.0,
                reference_pair={0: 100},
                hertz=10,
            )
            # print(df.head())
            # Saving norm df as csv
            new_csv = csv_path.replace(".csv", "baseline-10_0_gauss1.5.csv")
            df_sorted.to_csv(new_csv, index=False)
            try:
                df_sorted = insert_time_index_to_df(
                    df_sorted, range_min=-10.0, range_max=10.0, step=0.1
                )
            except ValueError:
                print("Index less than 200 data points!")
                df_sorted = insert_time_index_to_df(
                    df_sorted, range_min=-10.0, range_max=9.9, step=0.1
                )

            # Create scatter plot here
            # print(df_sorted.head())

            heatmap(
                df_sorted,
                csv_path,
                out_path=csv_path.replace(
                    ".csv", "_hm_baseline-10_0_gauss1.5.png"),
                vmin=-2.5,
                vmax=2.5,
                xticklabels=20,
            )

            spaghetti_plot(
                df_sorted,
                csv_path,
                out_path=csv_path.replace(
                    ".csv", "_spaghetti_baseline-10_0_gauss1.5.png"
                ),
            )
        except FileNotFoundError:
            print(f"File {csv_path} was not found!")
            pass

def get_max_of_df(df: pd.DataFrame):
    global_max = 0
    max_vals = list(df.max())

    for i in max_vals:
        if i > global_max:
            global_max = i
 
    return global_max

def get_min_of_df(df: pd.DataFrame):
    global_min = 9999999
    min_vals = list(df.min())

    for i in min_vals:
        if i < global_min:
            global_min = i
 
    return global_min

def new_main():

    ROOT_PATH = r"/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData"
    # ROOT_PATH = r"/Users/rodrigosandon/Documents/GitHub/LBGN/SampleData/truncating_bug"

    to_look_for = "all_concat_cells_choice_aligned_resampled_z_fullwindow.csv"


    csv_list = find_paths_endswith(
        ROOT_PATH, to_look_for
    )
    # print(csv_list)
    # csv_list.reverse()
    for count, csv_path in enumerate(csv_list):

        print(f"Working on file {count}: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
            # df = truncate_past_len_threshold(df, len_threshold=200)

            df = change_cell_names(df)

            # print(df.head())
            # We're essentially gettin the mean of z-score for a time frame to sort
            df_sorted = sort_cells(
                df,
                unknown_time_min=0.0,
                unknown_time_max=5.0,
                reference_pair={0: 100},
                hertz=10,
            )
            #print(list(df_sorted.columns))
            # print(df.head())
            # Saving norm df as csv
            """new_csv = csv_path.replace(
                ".csv", "baseline-10_0_gauss1.5_z_avgs.csv")
            df_sorted.to_csv(new_csv, index=False)"""
            #try:
            df_sorted = insert_time_index_to_df(
                df_sorted, range_min=-10.0, range_max=9.9, step=0.1
            )
            """ ValueError:
                print("Index less than 200 data points!")
                df_sorted = insert_time_index_to_df(
                    df_sorted, range_min=-10.0, range_max=9.9, step=0.1
                )"""

            # Create scatter plot here
            # print(df_sorted.head())
            max = get_max_of_df(df_sorted)
            min = get_min_of_df(df_sorted)

            heatmap(
                df_sorted,
                csv_path,
                out_path=csv_path.replace(
                    ".csv", "_hm.png"),
                vmin=min,
                vmax=max,
                xticklabels=20,
            )

            spaghetti_plot(
                df_sorted,
                csv_path,
                out_path=csv_path.replace(
                    ".csv", "_spaghetti.png"
                ),
            )
        except Exception as e:
            print(e)
            pass


def shock():

    ROOT_PATH = (
        r"/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/Shock Test"
    )
    # ROOT_PATH = r"/Users/rodrigosandon/Documents/GitHub/LBGN/SampleData/truncating_bug"

    csv_list = find_paths_startswith(ROOT_PATH, "all_concat_cells.csv")
    # print(csv_list)
    # csv_list.reverse()
    for count, csv_path in enumerate(csv_list):

        print(f"Working on file {count}: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
            # df = truncate_past_len_threshold(df, len_threshold=200)

            df = change_cell_names(df)

            # 3/4/2022 change: I want to see unorm heatmap
            df = custom_standardize(
                df,
                unknown_time_min=-5.0,
                unknown_time_max=0.0,
                reference_pair={0: 50},
                hertz=10,
            )

            df = gaussian_smooth(df.T)
            df = df.T
            # print(df.head())
            # We're essentially gettin the mean of z-score for a time frame to sort
            df_sorted = sort_cells(
                df,
                unknown_time_min=0.0,
                unknown_time_max=3.0,
                reference_pair={0: 50},
                hertz=10,
            )
            # print(df.head())
            df_sorted = insert_time_index_to_df(
                df_sorted, range_min=-5.0, range_max=5.0, step=0.1
            )

            # Create scatter plot here
            # print(df_sorted.head())

            heatmap(
                df_sorted,
                csv_path,
                out_path=csv_path.replace(".csv", "_hm.png"),
                vmin=-2.5,
                vmax=2.5,
                xticklabels=10,
            )

            spaghetti_plot(
                df_sorted,
                csv_path,
                out_path=csv_path.replace(".csv", "_spaghetti.png"),
            )
        except FileNotFoundError:
            print(f"File {csv_path} was not found!")
            pass


def shock_multiple_customs():

    ROOT_PATH = (
        r"/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/Shock Test"
    )
    # ROOT_PATH = r"/Users/rodrigosandon/Documents/GitHub/LBGN/SampleData/truncating_bug"

    csv_list = find_paths_startswith(ROOT_PATH, "all_concat_cells.csv")
    # print(csv_list)
    # csv_list.reverse()
    for count, csv_path in enumerate(csv_list):

        print(f"Working on file {count}: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
            # df = truncate_past_len_threshold(df, len_threshold=200)

            df = change_cell_names(df)

            # 3/4/2022 change: I want to see unorm heatmap
            # 3/10/22 : I wanna see two different z score baseline at once
            #print(df.iloc[70:90, :])
            df = custom_standardize_limit_fixed(
                df,
                baseline_min=0,
                baseline_max=20,
                limit_idx=40
            )
            #print(df.iloc[70:90, :])
            df = custom_standardize_limit_fixed(
                df,
                baseline_min=40,
                baseline_max=60,
                limit_idx=80
            )
            #print(df.iloc[70:90, :])
            df = gaussian_smooth(df.T)
            df = df.T
            # print(df.head())
            # We're essentially gettin the mean of z-score for a time frame to sort
            df_sorted = sort_cells(
                df,
                unknown_time_min=0.0,
                unknown_time_max=3.0,
                reference_pair={0: 50},
                hertz=10,
            )
            # print(df.head())
            df_sorted = insert_time_index_to_df(
                df_sorted, range_min=-6.0, range_max=6, step=0.1
            )
            """print(list(df_sorted.index)[0])
            print(list(df_sorted.index)[20])
            print(list(df_sorted.index)[40])
            print(list(df_sorted.index)[60])
            print(list(df_sorted.index)[80])"""
            # Create scatter plot here
            # print(df_sorted.head())
            # If you only want to show a subwindow of the subwindow
            # print(df_sorted.head())
            # print(list(df_sorted.index))
            df_sorted = df_sorted.iloc[0:81, :]

            heatmap(
                df_sorted,
                csv_path,
                out_path=csv_path.replace(".csv", "_double_z_hm2.png"),
                vmin=-2.5,
                vmax=2.5,
                xticklabels=10,
            )

            spaghetti_plot(
                df_sorted,
                csv_path,
                out_path=csv_path.replace(".csv", "_double_z_spaghetti2.png"),
            )
        except FileNotFoundError:
            print(f"File {csv_path} was not found!")
            pass


def process_one_table():

    csv_path = r"/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/RDT D1/Block_Reward Size_Shock Ocurred_Choice Time (s)/(1.0, 'Small', False)/all_concat_cells_z_fullwindow.csv"
    df = pd.read_csv(csv_path)
            # df = truncate_past_len_threshold(df, len_threshold=200)

    df = change_cell_names(df)

    # print(df.head())
    # We're essentially gettin the mean of z-score for a time frame to sort

    #CUSTOM REORDER HERE
    df_sorted = df[['1_C06', '1_C11', '1_C09', '14_C11', '15_C19', '16_C16', '3_C10', '13_C09', '16_C15', '16_C04', '1_C18', '1_C04', '16_C10', '13_C03', '3_C09', '16_C07', '6_C15', '11_C07', '15_C11', '14_C06', '1_C13', '16_C13', '6_C08', '6_C14', '16_C11', '16_C12', '19_C04', '19_C07', '15_C33', '15_C13', '3_C01', '15_C16', '15_C27', '19_C01', '15_C06', '16_C01', '11_C04', '6_C01', '16_C03', '3_C16', '13_C08', '11_C02', '15_C17', '16_C05', '6_C02', '6_C03', '6_C05', '11_C05', '15_C12', '13_C07', '6_C18', '8_C06', '14_C05', '1_C17', '15_C22', '6_C19', '16_C14', '6_C06', '14_C04', '9_C05', '15_C34', '1_C03', '15_C26', '11_C01', '3_C04', '3_C13', '14_C09', '3_C07', '9_C10', '14_C02', '9_C07', '9_C02', '15_C04', '15_C05', '9_C04', '3_C11', '15_C09', '19_C06', '6_C04', '15_C24', '7_C05', '8_C12', '7_C02', '11_C03', '1_C10', '8_C10', '13_C06', '11_C06', '15_C01', '15_C08', '13_C01', '19_C09', '16_C02', '15_C21', '15_C29', '15_C07', '15_C03', '8_C05', '1_C12', '1_C02', '6_C13', '15_C30', '15_C18', '8_C02', '9_C09', '3_C12', '13_C05', '14_C08', '9_C08', '9_C03', '8_C03', '14_C01', '1_C07', '15_C31', '3_C06', '6_C09', '1_C20', '7_C04', '3_C14', '8_C09', '3_C15', '19_C02', '9_C01', '8_C11', '6_C07', '7_C03', '8_C04', '15_C20', '9_C06', '14_C10', '15_C14', '16_C09', '15_C28', '14_C07', '15_C02', '16_C08', '6_C11', '15_C15', '1_C19', '3_C02', '1_C14', '16_C06', '15_C32', '14_C03', '3_C05', '13_C02', '3_C03', '19_C03', '19_C05', '8_C01', '15_C25', '15_C10', '6_C10', '3_C08', '1_C15', '1_C05', '13_C04', '1_C01', '15_C23', '1_C08', '1_C16', '6_C16', '8_C08', '19_C08', '6_C17', '15_C35', '1_C21', '6_C12', '8_C07', '7_C01']]
    """df_sorted = sort_cells(
        df,
        unknown_time_min=0.0,
        unknown_time_max=5.0,
        reference_pair={0: 100},
        hertz=10,
    )
    print(list(df_sorted.columns))"""

    df_sorted = insert_time_index_to_df(
        df_sorted, range_min=-10.0, range_max=10.0, step=0.1
    )

   
    max = get_max_of_df(df_sorted)
    print(max)
    min = get_min_of_df(df_sorted)
    print(min)

    #global
    """vmin=-2.805674410020901,
        vmax=5.189523721386852,"""

    heatmap(
        df_sorted,
        csv_path,
        out_path=csv_path.replace(
            ".csv", "_hm.svg"),
        vmin=-2.4064186234948832,
        vmax=3.49081465391537,
        xticklabels=20,
    )

    heatmap(
        df_sorted,
        csv_path,
        out_path=csv_path.replace(
            ".csv", "_hm.png"),
        vmin=-2.4064186234948832,
        vmax=3.49081465391537,
        xticklabels=20,
    )

def shock_one_mouse():

    ROOT_PATH = r"/media/rory/Padlock_DT/BLA_Analysis/"
    # ROOT_PATH = r"/Users/rodrigosandon/Documents/GitHub/LBGN/SampleData/truncating_bug"

    csv_list = find_paths(ROOT_PATH, "Shock Test", "concat_cells.csv")
    # print(csv_list)
    # csv_list.reverse()
    for count, csv_path in enumerate(csv_list):

        print(f"Working on file {count}: {csv_path}")

        df = pd.read_csv(csv_path)
        # df = truncate_past_len_threshold(df, len_threshold=200)

        df = change_cell_names(df)

        # 3/4/2022 change: I want to see unorm heatmap
        df = custom_standardize(
            df,
            unknown_time_min=-5.0,
            unknown_time_max=0.0,
            reference_pair={0: 50},
            hertz=10,
        )

        df = gaussian_smooth(df.T)
        df = df.T
        # print(df)
        # We're essentially gettin the mean of z-score for a time frame to sort
        df_sorted = sort_cells(
            df,
            unknown_time_min=0.0,
            unknown_time_max=3.0,
            reference_pair={0: 50},
            hertz=10,
        )
        # print(df.head())
        df_sorted = insert_time_index_to_df(
            df_sorted, range_min=-5.0, range_max=5.0, step=0.1
        )

        # Create scatter plot here
        # print(df_sorted.head())

        heatmap(
            df_sorted,
            csv_path,
            out_path=csv_path.replace(".csv", "_hm.png"),
            vmin=-2.5,
            vmax=2.5,
            xticklabels=10,
        )

        spaghetti_plot(
            df_sorted, csv_path, out_path=csv_path.replace(
                ".csv", "_spaghetti.png"),
        )


if __name__ == "__main__":
    #new_main()
    # shock()
    process_one_table()
    # shock_one_mouse()
    # shock_multiple_customs()
