from tslearn.metrics import dtw
from matplotlib.patches import ConnectionPatch
import scipy.spatial.distance as dist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from itertools import combinations
import time
from typing import List, Optional
import seaborn as sns


def dp(dist_mat):
    """
    Find minimum-cost path through matrix `dist_mat` using dynamic programming.

    The cost of a path is defined as the sum of the matrix entries on that
    path. See the following for details of the algorithm:

    - http://en.wikipedia.org/wiki/Dynamic_time_warping
    - https://www.ee.columbia.edu/~dpwe/resources/matlab/dtw/dp.m

    The notation in the first reference was followed, while Dan Ellis's code
    (second reference) was used to check for correctness. Returns a list of
    path indices and the cost matrix.
    """

    N, M = dist_mat.shape

    # Initialize the cost matrix
    cost_mat = np.zeros((N + 1, M + 1))
    for i in range(1, N + 1):
        cost_mat[i, 0] = np.inf
    for i in range(1, M + 1):
        cost_mat[0, i] = np.inf

    # Fill the cost matrix while keeping traceback information
    traceback_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            penalty = [
                cost_mat[i, j],  # match (0)
                cost_mat[i, j + 1],  # insertion (1)
                cost_mat[i + 1, j],
            ]  # deletion (2)
            i_penalty = np.argmin(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty

    # Traceback from bottom right
    i = N - 1
    j = M - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        tb_type = traceback_mat[i, j]
        if tb_type == 0:
            # Match
            i = i - 1
            j = j - 1
        elif tb_type == 1:
            # Insertion
            i = i - 1
        elif tb_type == 2:
            # Deletion
            j = j - 1
        path.append((i, j))

    # Strip infinity edges from cost_mat before returning
    cost_mat = cost_mat[1:, 1:]
    return (path[::-1], cost_mat)


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


def zscore(obs_value, mu, sigma):
    return (obs_value - mu) / sigma


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


def gaussian_smooth(df, sigma: float = 1.5):
    from scipy.ndimage import gaussian_filter1d

    return df.apply(gaussian_filter1d, sigma=sigma, axis=0)


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
        ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=5)
        ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=5)
        ax.tick_params(left=True, bottom=True)

        plt.title("Similarity Map")
        plt.savefig(out_path)
        plt.close()

    except ValueError as e:

        print("VALUE ERROR:", e)
        print(f"VALUE ERROR FOR {file_path} --> MAKING HEATMAP")
        pass


def change_cell_names(df):

    for col in df.columns:

        df = df.rename(columns={col: col.replace("BLA-Insc-", "")})
        # print(col)

    return df


def fill_points_for_hm(df):
    transposed_df = df.transpose()
    print(df.head())
    print(transposed_df.head())

    for row in list(transposed_df.columns):
        for col in list(transposed_df.columns):
            if int(transposed_df.loc[row, col]) == 0:
                transposed_df.loc[row, col] = df.loc[row, col]

    return df


def main():

    CONCAT_CELLS_PATH = r"/Users/rodrigosandon/Documents/GitHub/LBGN/SampleData/all_concat_cells.csv"
    start = time.time()

    print(f"Currently working on ... {CONCAT_CELLS_PATH}")
    df = pd.read_csv(CONCAT_CELLS_PATH)

    df = change_cell_names(df)

    # df = df.reindex(index=df.index[::-1])

    df = custom_standardize(
        df,
        unknown_time_min=-10.0,
        unknown_time_max=-1.0,
        reference_pair={0: 100},
        hertz=10,
    )

    df = gaussian_smooth(df.T)
    df = df.T

    to_select = 2  # can only compare two cells at a time
    cells_list = list(df.columns)

    combos = list(combinations(cells_list, to_select))
    # print(len(combos))

    # SETUP SKELETON DATAFRAME
    col_number = len(list(df.columns))
    cell_hm = pd.DataFrame(
        data=np.zeros((col_number, col_number)),
        index=list(df.columns),
        columns=list(df.columns),
    )
    # print("Skeleton df: ")
    # print(cell_hm)

    for count, combo in enumerate(combos):
        print(f"Working on combo {count}/{len(combos)}: {combo}")

        cell_x = list(combo)[0]
        cell_y = list(combo)[1]

        x = np.array(list(df[cell_x]))
        y = np.array(list(df[cell_y]))

        N = x.shape[0]
        M = y.shape[0]

        dist_mat = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                dist_mat[i, j] = abs(x[i] - y[j])

        path, cost_mat = dp(dist_mat)

        alignment_cost = cost_mat[N - 1, M - 1]
        norm_alignment_cost = cost_mat[N - 1, M - 1] / (N + M)

        """
        Overlaps: comparing cell 4 w/ cell 5 and cell 5 w/ cell 4 -> already covered by combo func
        Account for: never will compare cell w/ itself

        Note: all_concat_cells.csv is already sorted
        The first cell introduces all the indices neccesary, except for it's own (cell 1, cell 1)

        """

        # <- tthe closer the is to zero, the more similar
        cell_hm.loc[cell_x, cell_y] = norm_alignment_cost

        # print("Alignment cost: {:.4f}".format(alignment_cost))
        print("Normalized alignment cost: {:.4f}".format(norm_alignment_cost))

    cell_hm = fill_points_for_hm(cell_hm)

    cell_hm.to_csv(
        "/Users/rodrigosandon/Documents/GitHub/LBGN/SampleData/sim_map_all_concat_cells.csv", index=False)

    heatmap(
        cell_hm,
        CONCAT_CELLS_PATH,
        out_path=CONCAT_CELLS_PATH.replace(".csv", "_sim_map_final.png"),
        vmin=0.5,
        vmax=0,
        xticklabels=2,
    )

    end = time.time()
    print(f"Time taken: {end - start}")


if __name__ == "__main__":
    main()


"""print("Way 2")
        x = list(df[cell_x])
        y = list(df[cell_y])

        start_1 = time.time()
        cost = dtw(x, y, global_constraint="sakoe_chiba", sakoe_chiba_radius=3)
        print("Alignment cost: {:.4f}".format(cost))

        end_1 = time.time()
        print(f"Time taken: {end_1 - start_1}")

        print("Way 3")

        start_2 = time.time()
        cost = dtw(x, y, global_constraint="itakura", itakura_max_slope=2.)
        print("Alignment cost: {:.4f}".format(cost))

        end_2 = time.time()
        print(f"Time taken: {end_2 - start_2}")"""
