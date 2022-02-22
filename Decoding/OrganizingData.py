import pandas as pd
import os, glob
from pathlib import Path


def find_paths_endswith(root_path, endswith) -> list:

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )

    return files


class EventDataset:
    def __init__(self, event_name: str, root_folders: list) -> None:
        self.event_name = event_name
        self.root_folders = root_folders
