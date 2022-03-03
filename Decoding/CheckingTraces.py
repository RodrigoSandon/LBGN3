import os
import glob
import pandas as pd
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt


def find_paths_startswith(root_path, startswith) -> List:

    files = glob.glob(
        os.path.join(root_path, "**", "%s*") % (startswith), recursive=True,
    )

    return files


def spaghetti_plot(df: pd.DataFrame, trial, out_path):
    df_no_idx = df.iloc[:, 1:]
    x = list(df_no_idx.columns)
    df = df.T
    try:
        number_cells = 0
        for cell in df.columns:
            if cell != "Cell":  # ignore first tranposed col
                # print("cell: ", cell)
                plt.plot(x, list(df[cell]), label=cell)
                number_cells += 1

        plt.title(f"{trial} Cell Ca2+ Traces (n={number_cells})")
        plt.xlabel("Time (s)")
        plt.ylabel("dF/F")
        plt.locator_params(axis="x", nbins=20)
        plt.savefig(out_path)
        plt.close()

    except ValueError as e:

        print("VALUE ERROR:", e)
        pass

# /media/rory/Padlock_DT/BLA_Analysis/Decoding/Arranged_Dataset/Shock Test/Shock/0.32-0.4/BLA-Insc-1/trial_3.csv


def main():
    ROOT_PATH = Path(
        r"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Arranged_Dataset/Shock Test")

    files = find_paths_startswith(ROOT_PATH, "trial")

    for csv in files:
        csv = Path(csv)
        parts = csv.parts
        print(f"Processing {csv}")
        df = pd.read_csv(csv)

        trial = parts[-1].replace(".csv", "")
        png_out = csv.replace(".csv", "_spaghetti.csv")
        spaghetti_plot(df, trial, png_out)
