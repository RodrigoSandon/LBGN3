import os
import glob
import pandas as pd
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt


def find_paths(root_path: Path, mouse, startswith) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, "**", mouse, f"{startswith}*"), recursive=True,
    )
    return files


def spaghetti_plot(df: pd.DataFrame, trial, out_path):
    df_no_idx = df.iloc[:, 1:]
    x = list(df_no_idx.columns)
    new_x = []
    for i in x:
        # print(type(i))
        if "-" in i:
            i = i.replace("-", "")
            i = -abs(round(float(i), 1))
            # print(type(i))
            new_x.append(i)
        else:
            i = abs(round(float(i), 1))
            new_x.append(i)
    print(new_x)
    df = df.T
    try:
        number_cells = 0
        plt.locator_params(tight=True, axis="x", nbins=5)
        for cell in df.columns:
            if cell != "Cell":  # ignore first tranposed col
                # print("cell: ", cell)
                # print(x)
                # print(list(df[cell])[1:])
                plt.plot(new_x, list(df[cell])[1:], label=cell)
                number_cells += 1

        plt.title(f"{trial} Cell Ca2+ Traces (n={number_cells})")
        plt.xlabel("Time (s)")
        plt.ylabel("dF/F")
        plt.savefig(out_path)
        plt.close()

    except ValueError as e:

        print("VALUE ERROR:", e)
        pass


# /media/rory/Padlock_DT/BLA_Analysis/Decoding/Arranged_Dataset/Shock Test/Shock/0.32-0.4/BLA-Insc-1/trial_3.csv


def main():
    ROOT_PATH = Path(
        r"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Arranged_Dataset/Shock Test"
    )

    mouse = "BLA-Insc-1"
    files = find_paths(ROOT_PATH, mouse, "trial")
    # print(files)

    for csv in files:
        csv_parts = Path(csv)
        parts = list(csv_parts.parts)
        print(f"Processing {csv}")
        df = pd.read_csv(csv)

        trial = parts[11].replace(".csv", "")
        png_out = csv.replace(".csv", "_spaghetti.png")
        spaghetti_plot(df, trial, png_out)


if __name__ == "__main__":
    main()
