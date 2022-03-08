import os, glob
import pandas as pd
from pathlib import Path
from typing import List


def find_paths_iglob(root_path: Path, middle: str, endswith: str):
    for i in glob.iglob(
        os.path.join(root_path, "**", middle, "**", endswith), recursive=True,
    ):
        print(i)


def find_paths_glob(root_path: Path, middle: str, endswith: str) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, "**", middle, "**", endswith), recursive=True,
    )
    return files


# /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/Shock Test/SingleCellAlignmentData/C01/Bin_Shock Time (s)/0-0.1/plot_ready.csv
def shock_dissect_path(path: Path):
    parts = list(path.parts)
    intensity = parts[11]
    mouse = parts[6]
    session = parts[7]
    cell = parts[9]
    return intensity, mouse, session, cell


def main():
    ROOT_PATH = Path(r"/media/rory/Padlock_DT/BLA_Analysis")
    DST_ROOT_PATH = Path(
        r"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Pre-Arranged_Dataset/"
    )

    files = find_paths_glob(ROOT_PATH, "Bin_Shock Time (s)", "plot_ready.csv")
    # print(*files, sep="\n")

    # Now have all the shock test plot_ready.csv files for all cells for all sessions for all mice
    for csv in files:
        csv = Path(csv)
        intensity, mouse, session, cell = shock_dissect_path(csv)
        print(f"Processing {csv}")
        df: pd.DataFrame
        df = pd.read_csv(csv)
        new_df = df.iloc[:, 1:]  # remove first col
        neg_timepoints = [
            -1 * float(i.replace("-", "")) for i in list(new_df.columns) if "-" in i
        ]
        pos_timepoints = [float(i) for i in list(new_df.columns) if "-" not in i]
        all_timepoints = neg_timepoints + pos_timepoints
        # print(all_timepoints)
        detected_noshock_idxs = []
        detected_shock_idxs = []

        # Change the value of zero!
        for idx, timepoint in enumerate(all_timepoints):
            if timepoint < -1000000:  # if less than neg million, then it's zero
                all_timepoints[idx] = 0

        for idx, timepoint in enumerate(all_timepoints):
            # this exactly from csv -2.000000000000014
            if timepoint >= -6 and timepoint <= -2.000000000000014:
                detected_noshock_idxs.append(idx)
            if timepoint >= -2.000000000000014 and timepoint <= 2:
                detected_shock_idxs.append(idx)

        # since i want to keep the trial num col, make sure to add +1 to all idxs, and insert 0 into both dfs
        detected_noshock_idxs = [0] + [i + 1 for i in detected_noshock_idxs]
        print(len(detected_noshock_idxs))
        detected_shock_idxs = [0] + [i + 1 for i in detected_shock_idxs]
        print(len(detected_shock_idxs))

        noshock_trials: pd.DataFrame
        shock_trials: pd.DataFrame

        noshock_trials = df.iloc[:, detected_noshock_idxs]
        shock_trials = df.iloc[:, detected_shock_idxs]

        # Make destination dirs
        # /media/rory/Padlock_DT/BLA_Analysis/Decoding/Arranged_Dataset/{session}/{outcome}/{intensity}/{mouse}/{cell}/plot_ready.csv
        # all will come from the same session "Shock Test" so dont worry
        dst_dir_noshock = os.path.join(
            DST_ROOT_PATH, session, "No Shock", intensity, mouse, cell
        )
        dst_dir_shock = os.path.join(
            DST_ROOT_PATH, session, "Shock", intensity, mouse, cell
        )
        os.makedirs(dst_dir_noshock, exist_ok=True)
        os.makedirs(dst_dir_shock, exist_ok=True)

        # Save dfs to csvs
        noshock_trials.to_csv(
            os.path.join(dst_dir_noshock, "plot_ready.csv"), index=False
        )
        shock_trials.to_csv(os.path.join(dst_dir_shock, "plot_ready.csv"), index=False)
        break


if __name__ == "__main__":
    main()
