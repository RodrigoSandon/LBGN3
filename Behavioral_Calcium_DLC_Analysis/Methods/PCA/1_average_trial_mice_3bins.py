import os, glob
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

from matplotlib import animation 
from IPython.display import HTML

def find_paths_endswith(root_path, endswith) -> list:

    files = glob.glob(
        os.path.join(root_path, endswith), recursive=True,
    )

    return files

def main():
    # dirs will vary for another question we would like to ask
    blocks = ["1.0", "2.0", "3.0"]
    rew = ["Large", "Small"]
    shock = ["False", "True"]
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
        "BLA-Insc-14",
        "BLA-Insc-15",
        "BLA-Insc-16",
        "BLA-Insc-18",
        "BLA-Insc-19"
    ]

    sessions = ["RDT D1", "RDT D2", "RDT D3"]

    for block in blocks:
        for r in rew:
            for s in shock:
                for mouse in mice:
                    for session in sessions:

                        ROOT = f"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Unnorm_3Bin_Arranged_Dataset_-3_5/{block}/{r}/{s}/{mouse}/{session}"

                        files = find_paths_endswith(ROOT, "trial_*.csv")

                        try:
                            columns = list(pd.read_csv(files[0]).columns)[1:]

                            d: dict[str, list] = {}

                            for csv in files:
                                parent = Path(csv).parent
                                df = pd.read_csv(csv)

                                for row_idx in range(len(df)):
                                    cell_name = str(df.iloc[row_idx, 0])
                                    trial_trace = list(df.iloc[row_idx, 1:])
                                    #print(trial_trace)
                                    if cell_name in d:
                                        d[cell_name].append(trial_trace)
                                    else:
                                        d[cell_name] = []
                                        d[cell_name].append(trial_trace)
                            
                            d_avg = {}
                            for cell in d: 
                                # averaging across each timepoint
                                column_avg = [sum(trial_list)/len(trial_list) for trial_list in zip(*d[cell])]
                                d_avg[cell] = column_avg

                            avg_df = pd.DataFrame.from_dict(d_avg, orient='index', columns = columns)
                            #print(avg_df.head())
                            #print(avg_df.shape)
                            avg_df.to_csv(os.path.join(parent, "trials_average.csv"))
                        except Exception as e:
                            print(e)
                            pass


    
if __name__ == "__main__":
    main()