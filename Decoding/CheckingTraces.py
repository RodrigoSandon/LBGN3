import os
import glob
import pandas as pd
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import scipy.stats as stats


def find_paths(root_path: Path, mouse, startswith) -> List[str]:
    files = glob.iglob(
        os.path.join(root_path, "**", mouse, f"{startswith}*.csv"), recursive=True,
    )
    return files


def zscore(obs_value, mu, sigma):
    return (obs_value - mu) / sigma


def create_subwindow_fixed(
    my_list: list
) -> list:

    subwindow = my_list[0:21]

    return subwindow


def custom_standardize_list_fixed(
    my_list: list
) -> list:
    norm_list: list
    norm_list = []

    subwindow = create_subwindow_fixed(my_list)
    mean = stats.tmean(subwindow)
    stdev = stats.tstd(subwindow)

    for i in my_list:
        z_value = zscore(i, mean, stdev)
        norm_list.append(z_value)

    return norm_list


def spaghetti_plot(df: pd.DataFrame, trial, out_path, norm: bool):
    df_no_idx = df.iloc[:, 1:]
    x = list(df_no_idx.columns)
    #new_x = []
    """for i in x:
        # print(type(i))
        if "-" in i:
            i = i.replace("-", "")
            i = -abs(round(float(i), 1))
            # print(type(i))
            if i > 100000:
                i = 0
            new_x.append(i)
        else:
            i = abs(round(float(i), 1))
            new_x.append(i)"""
    # print(new_x)
    df = df.T
    try:
        number_cells = 0
        #plt.locator_params(tight=True, axis="x", nbins=5)
        for cell in df.columns:
            if cell != "Cell":  # ignore first tranposed col
                # print("cell: ", cell)
                # print(x)
                # print(list(df[cell])[1:])
                if norm == True:
                    plt.plot(x, custom_standardize_list_fixed(
                        list(df[cell])[1:]), label=cell)
                else:
                    plt.plot(x, list(df[cell])[1:], label=cell)
                number_cells += 1

        plt.title(f"{trial} Cell Ca2+ Traces (n={number_cells})")
        plt.xlabel("Time (s)")
        if norm == True:
            plt.ylabel("Z-score")
        else:
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
    ]
    for mouse in mice:
        files = find_paths(ROOT_PATH, mouse, "trial")
        # print(files)

        for csv in files:
            csv_parts = Path(csv)
            parts = list(csv_parts.parts)
            print(f"Processing {csv}")
            df = pd.read_csv(csv)
            norm = True

            trial = parts[11].replace(".csv", "")
            if norm == True:
                png_out = csv.replace(".csv", "_norm_spaghetti.png")
            else:
                png_out = csv.replace(".csv", "_spaghetti.png")
            spaghetti_plot(df, trial, png_out, norm)


if __name__ == "__main__":
    main()
