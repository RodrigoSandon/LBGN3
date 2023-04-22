import os
import os.path as path
import pandas as pd


def walk(top, topdown=True, onerror=None, followlinks=False, maxdepth=None):
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


session_count = 0
"""for root, dirs, files in walk("/media/rory/Padlock_DT/BLA_Analysis", maxdepth=4):
    print(root)"""

df_path = "/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/Late Shock D2/Block_Omission_Choice Time (s)/(1.0, 'ITI')/all_concat_cells.csv"

df = pd.read_csv(df_path)
print(df.tail())
print(len(df))
empty_val = df.iloc[200, df.columns.get_loc("BLA-Insc-2_C04")]
string_mode = str(df.iloc[200, df.columns.get_loc("BLA-Insc-2_C04")])
empty_val_type = type(df.iloc[200, df.columns.get_loc("BLA-Insc-2_C04")])
print(empty_val, string_mode, empty_val_type)
