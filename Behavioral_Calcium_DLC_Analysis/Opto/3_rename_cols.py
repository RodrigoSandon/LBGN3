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

#/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis/BLA_NAcShell/ArchT/Choice/RRD16/body/AlignmentData/Block_Trial_Type_Reward_Size_Start_Time_(s)/(1.0, 'Forced', 'Large')/speeds_z_-5_5savgol_avg.csv
def main():


    session_root = r"/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis/BLA_NAcShell/ArchT/choice/RRD76"

    """list_of_combos_we_care_about = [
            "Block_Start_Time_(s)",
            "Block_Omission_Start_Time_(s)",
            "Block_Reward_Size_Start_Time_(s)",
            "Block_Reward_Size_Shock_Ocurred_Start_Time_(s)",
            "Block_Shock_Ocurred_Start_Time_(s)",
            "Block_Trial_Type_Start_Time_(s)",
            "Shock_Ocurred_Start_Time_(s)",
            "Trial_Type_Start_Time_(s)",
            "Trial_Type_Reward_Size_Start_Time_(s)",
            "Block_Trial_Type_Omission_Start_Time_(s)",
            "Block_Trial_Type_Reward_Size_Start_Time_(s)",
            "Block_Trial_Type_Shock_Ocurred_Start_Time_(s)",
            "Block_Trial_Type_Win_or_Loss_Start_Time_(s)",
            "Trial_Type_Shock_Ocurred_Start_Time_(s)",
            "Win_or_Loss_Start_Time_(s)",
            "Block_Win_or_Loss_Start_Time_(s)",
            "Learning_Stratergy_Start_Time_(s)",
            "Omission_Start_Time_(s)",
            "Reward_Size_Start_Time_(s)",
        ]"""

    list_of_combos_we_care_about = [

            "Block_Trial_Type_Start_Time_(s)",
        ]

    filename = "speeds_z_-5_5savgol_avg.csv"

    for combo in list_of_combos_we_care_about:

        files = find_paths(session_root, combo ,filename)


        for csv in files:
            mouse = csv.split("/")[9]

            df = pd.read_csv(csv)

            #df = df.rename(columns={f"{mouse}_Avg_Speed_(cm/s)" : f"Avg_Speed_(cm/s)"})
            df = df.rename(columns={f"Avg_Speed_(cm/s)" : f"{mouse}_Avg_Speed_(cm/s)"})
            
            df.to_csv(csv, index=False)

if __name__ == "__main__":
    main()