import pandas as pd
import numpy as np
import os, glob, shutil
from pathlib import Path

def find_paths_startswith(root, startswith) -> list:

    files = glob.glob(os.path.join(root, "**", f"{startswith}*"), recursive=True)

    return files

root = "/media/rory/RDT VIDS/BORIS_merge/RRD113"
files = find_paths_startswith(root, "rrd")
files.reverse()

print(files)
for file in files:
    os.rename(file, file.replace("rrd", "RRD"))