import matplotlib.pyplot as plt
from typing import List, Optional
import pandas as pd
import seaborn as sns
from operator import attrgetter

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
            f"Trial Z-Score Traces (n={len(list(df.columns))})"
        )
        plt.savefig(out_path)
        plt.close()

    except ValueError as e:

        print("VALUE ERROR:", e)
        print(f"VALUE ERROR FOR {file_path} --> MAKING HEATMAP")
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

    def convert_lst_to_d(lst):
        res_dct = {}
        for count, i in enumerate(lst):
            res_dct[i.cell_name] = i.dff_traces

        print(f"NUMBER OF CELLS: {len(lst)}")
        return res_dct

    sorted_cells_d = convert_lst_to_d(sorted_cells)

    df_mod = pd.DataFrame.from_dict(
        sorted_cells_d
    )  
    # from_records automatically sorts by key
    # from_dict keeps original order intact
    # print(df_mod)
    return df_mod

def main():
    example = ""

if __name__ == "__main__":
    main()