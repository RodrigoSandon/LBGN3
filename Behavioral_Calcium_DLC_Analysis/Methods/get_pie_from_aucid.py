from ast import List
import glob
import os
from pathlib import Path
import pandas as pd
import numpy as np
import math
import random

from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt
from operator import attrgetter
from scipy import stats

def change_cell_names(df):

    for col in df.columns:

        df = df.rename(columns={col: col.replace("BLA-Insc-", "")})
        # print(col)

    return df

def pie_chart(
    csv_path: str, test_name: str, data: list, labels: list, replace_name: str
):
    fig = plt.figure(figsize=(10, 7))
    plt.pie(data, labels=labels, autopct="%1.2f%%")
    plt.title(test_name)
    new_name = csv_path.replace(".csv", replace_name)
    plt.savefig(new_name)
    plt.close()


class CellClassification():
    def __init__(
        self,
        csv_path: str,
        df: pd.DataFrame,
        test: str,
    ):

        self.csv_path = csv_path
        self.df = df
            
        if test == "wilcoxon rank sum test":
            CellClassification.wilcoxon_rank_sum(self)


    def wilcoxon_rank_sum(self):  # wilcoxon rank sum test

        active_cells = []
        inactive_cells = []
        neutral_cells = []

        number_cells = len(list(self.df.columns))

        for col in list(self.df.columns):  # a col is a cell
            if list(self.df[col])[0] == "+":
                active_cells.append(col)
            elif list(self.df[col])[0] == "-":
                inactive_cells.append(col)
            else:
                neutral_cells.append(col)
            

        d = {
            "(+) Active Cells": len(active_cells),
            "(-) Active Cells": len(inactive_cells),
            "Neutral Cells": len(neutral_cells),
        }

        pie_chart(
            self.csv_path,
            f"Wilcoxon Rank Sum Test (n={number_cells})",
            list(d.values()),
            list(d.keys()),
            replace_name=f"fullwindow_z_id_auc_manwhitney_pie.png",
        )

def find_paths(root_path: Path, middle: str, endswith: str):
    files = glob.glob(
        os.path.join(root_path, "**", middle, "**", endswith), recursive=True,
    )
    return files
    
def main_allcells():

    ROOT = r"/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/"
    sessions = ["RDT D1", "RDT D2", "RDT D3"]

    for session in sessions: 
        files = find_paths(ROOT, session, "all_concat_cells_z_fullwindow_id_auc.csv")

        for f in files:
            df = pd.read_csv(f)
            df = change_cell_names(df)

            CellClassification(
                f,
                df,
                test="wilcoxon rank sum test",
            )

def main_permouse():

    MASTER_ROOT = r"/media/rory/Padlock_DT/BLA_Analysis"
    mice = [
        "BLA-Insc-1",
        "BLA-Insc-2",
        "BLA-Insc-3",
        "BLA-Insc-5",
        "BLA-Insc-6",
        "BLA-Insc-7",
        "BLA-Insc-8",
        "BLA-Insc-9",
        "BLA-Insc-11",
        "BLA-Insc-13",
        "BLA-Insc-14",
        "BLA-Insc-15",
        "BLA-Insc-16",
        "BLA-Insc-18",
        "BLA-Insc-19",
        ]
    sessions = ["RDT D1", "RDT D2", "RDT D3"]
    event = "Shock Ocurred_Choice Time (s)"
    # /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-11/RDT D2/BetweenCellAlignmentData/Block_Win or Loss_Choice Time (s)/(3.0, 'Loss')/concat_cells.csv
    for mouse in mice:
        for session in sessions:
            files = find_paths(MASTER_ROOT, f"{mouse}/{session}/BetweenCellAlignmentData/{event}","concat_cells_id_fullwindow_z_auc.csv")
            for f in files:
                df = pd.read_csv(f)
                df = change_cell_names(df)

                CellClassification(
                    f,
                    df,
                    test="wilcoxon rank sum test",
                )


if __name__ == "__main__":
    main_allcells()
    #main_permouse()
