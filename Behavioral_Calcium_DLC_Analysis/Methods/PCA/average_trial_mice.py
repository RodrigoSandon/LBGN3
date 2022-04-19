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
            for mouse in mice:
                for session in sessions:

                    ROOT = f"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Arranged_Dataset/{block}/{r}/{mouse}/{session}"

                    files = find_paths_endswith(ROOT, "trial_*.csv")

                    #print(files)
                    # cell: list of lists
                    try:
                        columns = list(pd.read_csv(files[0]).columns)[1:]
                        #index = list(pd.read_csv(files[0]).iloc[:, 0])

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


                    """# Run PCA for each of the timepoints (the z scores are averaged across trials)

                    for time_point in avg_df:
                        subset_df = avg_df[[time_point]]
                        #print(subset_df.head())
                        pca_obj = PCA()

                        pca_obj.fit(subset_df)
                        pca_data = pca_obj.transform(subset_df)
                        per_var = np.round(pca_obj.explained_variance_ratio_*100, decimals=1)
                        labels = ['PC' + str(x) for x in range(1,len(per_var)+1)]
                        neurons = list(avg_df.index)

                        pca_df = pd.DataFrame(pca_data, index=neurons, columns = labels)
                        print(pca_df.head())

                        plt.plot(pca_df.PC1)
                        plt.title("PCA Graph")
                        plt.xlabel("Time (s)")
                        plt.ylabel("PC1 - {0}%".format(per_var[0]))

                        for sample in pca_df.index:
                            #print(type(pca_df.PC1.loc[sample]))
                            plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))

                        plt.show()"""


    
if __name__ == "__main__":
    main()