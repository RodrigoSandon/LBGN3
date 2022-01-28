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


def find_paths_endswith(root_path, endswith) -> List:

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith),
        recursive=True,
    )

    return files


def find_paths_startswith(root_path, startswith) -> List:

    files = glob.glob(
        os.path.join(root_path, "**", "%s*") % (startswith),
        recursive=True,
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

        print(f"NUMBER OF CELLS: {len(lst)}")
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
        plt.title("Smoothed Z-Scores of Neural Ca2+ Traces (n=%s)" % (number_cells))
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
    df, unknown_time_min, unknown_time_max, reference_pair: dict, hertz: int
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

    x_axis[middle_idx] = 0
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
                unknown_time_max=-1.0,
                reference_pair={0: 100},
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
                reference_pair={0: 100},
                hertz=10,
            )
            # print(df.head())
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
                out_path=csv_path.replace(".csv", "_hm_baseline-10_-1_gauss1.5.png"),
                vmin=-2.5,
                vmax=2.5,
                xticklabels=20,
            )

            spaghetti_plot(
                df_sorted,
                csv_path,
                out_path=csv_path.replace(
                    ".csv", "_spaghetti_baseline-10_-1_gauss1.5.png"
                ),
            )
        except FileNotFoundError:
            print(f"File {csv_path} was not found!")
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
                out_path=csv_path.replace(".csv", "_hm_baseline-5_0_gauss1.5.png"),
                vmin=-2.5,
                vmax=2.5,
                xticklabels=10,
            )

            spaghetti_plot(
                df_sorted,
                csv_path,
                out_path=csv_path.replace(
                    ".csv", "_spaghetti_baseline-5_0_gauss1.5.png"
                ),
            )
        except FileNotFoundError:
            print(f"File {csv_path} was not found!")
            pass


if __name__ == "__main__":
    # main()
    shock()


def process_one_table():

    csv_path = r"/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/RDT D2/Shock Ocurred_Choice Time (s)/True/all_concat_cells.csv"
    df = pd.read_csv(csv_path)
    # df = truncate_past_len_threshold(df, len_threshold=200)

    df = change_cell_names(df)
    # print("AFTER NAME CHANGE:")
    # print(df.head())

    df = custom_standardize(
        df,
        unknown_time_min=-10.0,
        unknown_time_max=-1.0,
        reference_pair={0: 100},
        hertz=10,
    )
    # print("AFTER STANDARDIZATION:")
    # print(df.head())

    # SMOOTHING NEEDS TRANSPOSE BC SMOOTHING RELATIVE TO EACH CELL'S BASELINE (so axis should be 1
    # bc smoothing across columns)
    df = gaussian_smooth(df.T)
    # THEN TRANSPOSE IT BACK
    df = df.T

    # print("AFTER SMOOTHING:")
    # print(df.head())

    df_sorted = sort_cells(
        df,
        unknown_time_min=0.0,
        unknown_time_max=3.0,
        reference_pair={0: 100},
        hertz=10,
    )

    # print("AFTER SORTING:")
    # print(df_sorted.head())
    subdf = subdf_of_df(
        df_sorted,
        unknown_time_min=0.0,
        unknown_time_max=3.0,
        reference_pair={0: 100},
        hertz=10,
    )

    # MAKE SURE NEW IDX IS LAST BEFORE PLOTS, MESSES UP PREVIOUS PREPROCESSING
    df_sorted_new_idx = insert_time_index_to_df(df_sorted)
    """print("AFTER INDEX CHANGE:")
    print(df_sorted.head())"""

    heatmap(
        df_sorted_new_idx,
        csv_path,
        out_path=csv_path.replace(".csv", "_sorted_hm_baseline-10_-1_gauss1.5.png"),
        vmin=-2.5,
        vmax=2.5,
        xticklabels=20,
    )

    spaghetti_plot(
        df_sorted_new_idx,
        csv_path,
        out_path=csv_path.replace(
            ".csv", "_sorted_spaghetti_baseline-10_-1_gauss1.5.png"
        ),
    )

    # Create SCATTER PLOT of subwindows of each cell's window of specific event
    # First create sub df -> bc we don't want to cluster based on the 200 data points only
    # by 30 data points

    # this scatter plot is plotting smoothed z-score of each cell's dff

    # print(subdf.head())

    scatter_plot(
        subdf,
        out_path=csv_path.replace(
            ".csv", "_scatter_window0_3s_baseline-10_-1_gauss1.5.png"
        ),
    )
