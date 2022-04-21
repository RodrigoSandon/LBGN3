import os, glob
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

from matplotlib import animation 
from IPython.display import HTML

def find_paths_endswith(root_path, endswith) -> list:

    files = glob.glob(
        os.path.join(root_path, "*%s") % (endswith), recursive=True,
    )

    return files

def pca_df(df: pd.DataFrame) :
    #df = pd.read_csv(csv)
    df = df.iloc[:, 1:]
    df = df.T
    arr = df.to_numpy()

    pca_obj = PCA()

    pca_obj.fit(arr)
    
    pca_data = pca_obj.transform(arr)
 
    per_var = np.round(pca_obj.explained_variance_ratio_*100, decimals=1)
    labels = ['PC' + str(x) for x in range(1,len(pca_data)+1)]
    print(len(labels))

    time = list(df.columns)

    pca_df = pd.DataFrame(pca_data.T, index=time, columns = labels)
    print(pca_df.head())
    

    return pca_df, per_var

def combiner(csvs_to_concat: list) -> pd.DataFrame:
    dfs_to_concat = [pd.read_csv(i) for i in csvs_to_concat]
    result = pd.concat(dfs_to_concat, axis=1)
    return result

def main():

    """/media/rory/Padlock_DT/BLA_Analysis/Decoding/Generalized_PCA_-3_5/1.0/Small/RDT D1/all_cells_avg_trials.csv",
        "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Generalized_PCA_-3_5/2.0/Small/RDT D1/all_cells_avg_trials.csv",
        "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Generalized_PCA_-3_5/3.0/Small/RDT D1/all_cells_avg_trials.csv"""

    """/media/rory/Padlock_DT/BLA_Analysis/Decoding/Generalized_PCA_-3_5/1.0/Large/RDT D1/all_cells_avg_trials.csv",
        "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Generalized_PCA_-3_5/2.0/Large/RDT D1/all_cells_avg_trials.csv",
        "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Generalized_PCA_-3_5/3.0/Large/RDT D1/all_cells_avg_trials.csv"""

    """/media/rory/Padlock_DT/BLA_Analysis/Decoding/Generalized_PCA_-3_5/1.0/Small/RDT D1/all_cells_avg_trials.csv",
    "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Generalized_PCA_-3_5/1.0/Large/RDT D1/all_cells_avg_trials.csv"""

    """/media/rory/Padlock_DT/BLA_Analysis/Decoding/Generalized_PCA_-3_5/2.0/Small/RDT D1/all_cells_avg_trials.csv",
    "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Generalized_PCA_-3_5/2.0/Large/RDT D1/all_cells_avg_trials.csv"""

    """/media/rory/Padlock_DT/BLA_Analysis/Decoding/Generalized_PCA_-3_5/3.0/Small/RDT D1/all_cells_avg_trials.csv",
    "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Generalized_PCA_-3_5/3.0/Large/RDT D1/all_cells_avg_trials.csv"""

    csvs_to_concat = [
        "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Generalized_PCA_-3_5/3.0/Small/RDT D1/all_cells_avg_trials.csv",
    "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Generalized_PCA_-3_5/3.0/Large/RDT D1/all_cells_avg_trials.csv"
        ]
    
    concatenated = combiner(csvs_to_concat)
    df, per_var = pca_df(concatenated)

    plt.scatter(df.PC1, df.PC2, label="Large")
    print(len(df.PC1))

    plt.legend()
    plt.title("PCA Graph")
    plt.xlabel(f"PC1 - {per_var[0]}")
    plt.ylabel(f"PC2 - {per_var[1]}")

    """for sample in pca_df_L.index:
        #print(type(pca_df.PC1.loc[sample]))
        plt.annotate(sample, (pca_df_L.PC1.loc[sample], pca_df_L.PC2.loc[sample]))
        plt.annotate(sample, (pca_df_S.PC1.loc[sample], pca_df_S.PC2.loc[sample]))"""

    
    plt.show()


    
if __name__ == "__main__":
    main()