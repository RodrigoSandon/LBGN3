import pandas as pd
import numpy as np
import os, glob, shutil
import os.path as path
from pathlib import Path

def find_files(root, filename) -> list:

    files = glob.glob(os.path.join(root, "**", f"{filename}"), recursive=True)

    return files


def walk(top, topdown=False, onerror=None, followlinks=False, maxdepth=None):
    islink, join, isdir = path.islink, path.join, path.isdir

    try:
        names = os.listdir(top)
    except OSError as err:
        if onerror is not None:
            onerror(err)
        return

    dirs, nondirs = [], []
    for name in names:
        if isdir(join(top, name)):
            dirs.append(name)
        else:
            nondirs.append(name)

    if topdown:
        yield top, dirs, nondirs

    if maxdepth is None or maxdepth > 1:
        for name in dirs:
            
            new_path = join(top, name)
            if followlinks or not islink(new_path):
                for x in walk(
                    new_path,
                    topdown,
                    onerror,
                    followlinks,
                    None if maxdepth is None else maxdepth - 1,
                ):
                    yield x
    if not topdown:
        yield top, dirs, nondirs

root_1 = r"/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis/"
sleap_filename = "body_sleap_data.csv"
session_type = "Choice"
# Not only do we wanna ignore outcomes when singularly processing the outcome file, but
# also when we're trying to concatenate across, so this prevents that

""" Concatenates sleap data files up to a certain point """

for root, dirs, files in walk(root_1, maxdepth=4):
    if session_type in root:
        print("CURR ROOT: ", root)
        # Set up for the dictionary you're about to make (concatenated speed cols)
        # Get an example of the col for idx time for the rest of the cols you're in
        example_sleap_file = pd.read_csv(f"/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis/BLA_NAcShell/ArchT/Choice/RRD16/body/{sleap_filename}")
        # with this example sleap file, just grab it's time col
        example_time_col = list(example_sleap_file["idx_time"])
        speed_cols_d = {
            "Time (s)" : example_time_col,
        }
        # Find all sleap files in this curr root   
        sleap_files = find_files(root, sleap_filename)
        # Go through each sleap file and open it
        for sleap_path in sleap_files:
            mouse_name = sleap_path.split("/")[9]
            print(mouse_name)

            sleap_file_df = pd.read_csv(sleap_path)
            # grab the speed column and put it in dict
            speed_col = list(sleap_file_df["vel_cm_s"])
            # name of col must include mouse name, the entire concat file should include i it's choice or not
            speed_cols_d[f"{mouse_name} (cm/s)"] = speed_col

            # Now that you have individual mouse speeds aligned, you're able to convert to df

        concat_speeds_df = pd.DataFrame.from_dict(speed_cols_d)
        print("Glance at concat df:")
        print(concat_speeds_df.head())
        # Where will you save? in the root folder provided by os.walk
        concat_filename = f"{session_type}_speeds_concat.csv"
        concat_csv_dst = os.path.join(root, concat_filename)
        concat_speeds_df.to_csv(concat_csv_dst, index=False)




