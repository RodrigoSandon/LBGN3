from typing import List
from pathlib import Path
import os
import glob
import os.path as path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def find_paths(root_path: Path, endswith: str) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, "**", endswith), recursive=True,
    )
    return files

def find_csv_files(session_path, startswith):

    files = glob.glob(
        os.path.join(session_path, "**", "%s*") % (startswith),
        recursive=True,
    )

    return files

def add_val(arr, val):
    
    return np.append(arr, [val])

def concat_trials_across_bins(
    lst_of_all_avg_concat_cells_path, root_path, filename
):
    between_mice_d = {}
    
    # /media/rory/RDT VIDS/BORIS/RRD170/RDT OPTO CHOICE 0115/AlignmentData/Block_Trial_Type_Reward_Size_Start_Time_(s)/(1.0, 'Forced', 'Large')/speeds_z_-5_5savgol_avg.csv
    # 0  1      2       3               4           5           [6]      [7]    [8]   [9]   10         11                       [12]                                  [13]              14
    # /media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis/vHPC_NAcShell/eYFP/Choice/RRD124/body/AlignmentData/Block_Trial_Type_Reward_Size_Start_Time_(s)/(1.0, 'Forced', 'Large')/avg_speed.csv
    for avg_concat_cells_csv_path in lst_of_all_avg_concat_cells_path:
        print(f"Currently working on ...{avg_concat_cells_csv_path}")

        circuit = avg_concat_cells_csv_path.split("/")[6]
        treatment = avg_concat_cells_csv_path.split("/")[7]
        
        ###PARAMS CATEGORIZED BY###
        if avg_concat_cells_csv_path.split("/")[8].lower() == "choice":

            session_type = "Choice"
            
        elif avg_concat_cells_csv_path.split("/")[8].lower() == "outcome":
            session_type = "Outcome"

        mouse_name = avg_concat_cells_csv_path.split("/")[9]
        combo = avg_concat_cells_csv_path.split("/")[12]
        subcombo = avg_concat_cells_csv_path.split("/")[13]

        # means nothing has been created yet
        if circuit not in between_mice_d:
            between_mice_d[circuit] = {}

        if treatment not in between_mice_d[circuit]:
            between_mice_d[circuit][treatment] = {}

        if session_type not in between_mice_d[circuit][treatment]:
            between_mice_d[circuit][treatment][session_type] = {}

        if combo not in between_mice_d[circuit][treatment][session_type]:
            between_mice_d[circuit][treatment][session_type][combo] = {}

        if subcombo not in between_mice_d[circuit][treatment][session_type][combo]:
            between_mice_d[circuit][treatment][session_type][combo][subcombo] = {}

        #print(between_mice_d)                    
        df = pd.read_csv(avg_concat_cells_csv_path)
        
        for col_name, col_data in df.iteritems():
            # for however many cols there are under this mouse, for this session type, combo, n subcombo
            # TIME WONT BE INCLUDED MORE THAN ONCE
            if col_name not in between_mice_d[circuit][treatment][session_type][combo][subcombo]:
                if len(df[col_name].tolist()) != 299:
                    print("key not included:", circuit, treatment, session_type, combo, subcombo, col_name)
                else:
                    between_mice_d[circuit][treatment][session_type][combo][subcombo][col_name] = df[col_name].tolist()

    for circuit in between_mice_d:
        for treatment in between_mice_d[circuit]:
            for session_type in between_mice_d[circuit][treatment]:
                for combo in between_mice_d[circuit][treatment][session_type]:
                    for subcombo in between_mice_d[circuit][treatment][session_type][combo]:
                        try:
                            concatenated_df = pd.DataFrame.from_dict(
                                between_mice_d[circuit][treatment][session_type][combo][subcombo]
                            )

                        except ValueError:
                            print("JAGGED ARRAYS IN:", circuit, treatment, session_type, combo, subcombo)   
                            continue

                        d = between_mice_d[circuit][treatment][session_type][combo][subcombo]
                        
                        concatenated_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))

                        new_path = os.path.join(root_path, circuit, treatment, session_type, combo, subcombo)
                        
                        os.makedirs(new_path, exist_ok=True)
                        concatenated_df.to_csv(os.path.join(new_path, f"all_{filename}"), index=False)
    


def main():
    ROOT_PATH = Path(r"/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis")

    filename = "speeds_z_-5_5savgol_renamed.csv"

    csv_paths = find_paths(ROOT_PATH, filename) #CUSTOMIZE FOR SPECIFIC GROUPINGS YOU WANT TO PROCESS
    dst_dir= "BetweenMiceAlignmentData"
    new_root_path = os.path.join(ROOT_PATH, dst_dir)
    
    
    concat_trials_across_bins(
        csv_paths, new_root_path, filename
        )


main()