import os, glob
from pathlib import Path
from typing import List
import pandas as pd

def find_paths(root_path: Path, endswith: str) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, "**", endswith), recursive=True,
    )
    return files

def main():
    ROOTS = [
        "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Pearson_Input_Datasets/Neural_Net/RDT D1",
        "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Pearson_Input_Datasets/Neural_Net/RDT D2",
        "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Pearson_Input_Datasets/Neural_Net/RDT D3",
        ]

    max_cell_corrmap = 0
    max_cell_description = ""

    min_cell_corrmap = 99999
    min_cell_description = ""
    for root in ROOTS:
        files = find_paths(root, "trial_*_corrmap_*.csv")

        for f in files:
            df = pd.read_csv(f)
            df = df.iloc[:, 1:]
            name = Path(f).name
            #print(list(df.columns))
            #print(len(list(df.columns)))
            num_cells = len(list(df.columns))
            if num_cells < min_cell_corrmap:
                min_cell_corrmap = num_cells
                min_cell_description = name

            if num_cells > max_cell_corrmap:
                max_cell_corrmap = num_cells
                max_cell_description = name
    
    print(f"Min cells in a corrmap: {min_cell_corrmap} -> {min_cell_description}")
    print(f"Max cells in a corrmap: {max_cell_corrmap} -> {max_cell_description}")
        

if __name__ == "__main__":
    main()