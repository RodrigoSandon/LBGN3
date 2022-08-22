import numpy as np
import glob
import os
from typing import List
import pandas as pd
from pathlib import Path

def find_paths(root_path: Path, middle: str, endswith: str) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, "**", middle, "**", endswith), recursive=True,
    )
    return files

#/media/rory/RDT VIDS/BORIS/RRD170/RDT OPTO CHOICE 0115/AlignmentData/Block_Trial_Type_Reward_Size_Start_Time_(s)/(1.0, 'Forced', 'Large')/speeds_z_-5_5savgol_avg.csv
def main():

    sessions = [
        "/media/rory/RDT VIDS/BORIS/RRD170/RDT OPTO CHOICE 0115",
        "/media/rory/RDT VIDS/BORIS/RRD168/RDT OPTO CHOICE 0114",
        "/media/rory/RDT VIDS/BORIS/RRD171/RDT OPTO CHOICE 0104",
        "/media/rory/RDT VIDS/BORIS/RRD81/RDT OPTO CHOICE 1104"
        ]
    
    # OR

    session_root = r"/media/rory/RDT VIDS/BORIS/"

    combo = "Block_Trial_Type_Reward_Size_Start_Time_(s)"

    for session in sessions:
        filename = "speeds_z_-5_5savgol_avg.csv"
        files = find_paths(session, f"{combo}",filename)

        for csv in files:
            mouse_session = "_".join(csv.split("/")[5:7])

            df = pd.read_csv(csv)

            df = df.rename(columns={"Avg_Speed_(cm/s)" : f"{mouse_session}_Avg_Speed_(cm/s)"})
            
            df.to_csv(csv, index=False)

if __name__ == "__main__":
    main()