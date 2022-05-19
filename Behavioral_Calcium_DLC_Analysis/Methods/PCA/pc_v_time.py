from cgitb import small
import os, glob
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

from matplotlib import animation 
from IPython.display import HTML

def find_paths_endswith(root_path, endswith) -> list:

    files = glob.glob(
        os.path.join(root_path, "*%s") % (endswith), recursive=True,
    )

    return files

def pca_df(df: pd.DataFrame) :
    #df = pd.read_csv(csv)
    #df = df.iloc[:, 1:]
    #df = df.T
    
    #print(df)

    pca_obj = PCA()

    pca_obj.fit(df)
    
    pca_data = pca_obj.transform(df)
 
    per_var = np.round(pca_obj.explained_variance_ratio_*100, decimals=1)
    labels = ['PC' + str(x) for x in range(1,len(per_var)+1)]
    #print(len(labels))

    time = list(df.T.columns)

    pca_df = pd.DataFrame(pca_data, index=time, columns = labels)
    #print(pca_df)
    

    return pca_df, per_var, labels

def combiner_same_block(csvs_to_concat: list) -> pd.DataFrame:
    # making empty arrays that will be filled with indicies later
    multi_cols = [[],[]] # type: List[List[str]]
    dfs_to_concat = []
  
    for csv in csvs_to_concat:
        df = pd.read_csv(csv)
        df = df.iloc[:, 1:]
        df = df.T
        dfs_to_concat.append(df)
        block = csv.split("/")[7]
        num_cols = len(list(df.columns))
        for count in range(num_cols):
            multi_cols[0].append(block)
        for col in list(df.columns):
            multi_cols[1].append(col)
    
    multi_cols = [np.asarray(mylist) for mylist in multi_cols]
    result = pd.concat(dfs_to_concat, axis=1)
    result.columns = multi_cols

    return result.T

def main():

    csvs_to_concat = [
        "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Unnorm_3Bin_Generalized_PCA_-3_5/1.0/Large/False/RDT D1/all_cells_avg_trials.csv",
        "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Unnorm_3Bin_Generalized_PCA_-3_5/2.0/Large/True/RDT D1/all_cells_avg_trials.csv",
        "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Unnorm_3Bin_Generalized_PCA_-3_5/3.0/Large/True/RDT D1/all_cells_avg_trials.csv"
        ]
    
    concatenated_df = combiner_same_block(csvs_to_concat)
    print("CONCATENATED DF BEFORE STANDARDIZATION")
    print(concatenated_df.head())
    #print(len(concatenated_df["Small"].columns))
    #print(len(concatenated_df["Large"].columns))

    ################## ZSCORE CONCATENATED DFS ##################
    def zscore(obs_value, mu, sigma):
        return (obs_value - mu) / sigma
    from scipy import stats

    mylist = []
    for col in list(concatenated_df.columns):
        for idx, row in enumerate(list(concatenated_df[col])):
            mylist.append(concatenated_df.iloc[idx][col])
    
    mean = stats.tmean(mylist)
    stdev = stats.tstd(mylist)

    for col in list(concatenated_df.columns):
        for idx, row in enumerate(list(concatenated_df[col])):
            concatenated_df.iloc[idx][col] = zscore(concatenated_df.iloc[idx][col], mean, stdev)

    ################## ZSCORE CONCATENATED DFS ##################
    print("CONCATENATED DF AFTER STANDARDIZATION")
    print(concatenated_df.head())
    

    df, per_var, labels = pca_df(concatenated_df)

    print("PCA DF")
    print(df.head())
    per_var = per_var[0:10]
    labels = labels[0:10]
    plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.show()
    #print(df)

    b1_idx = [i for i in list(df.index) if "1.0" in list(i)]
    b2_idx = [i for i in list(df.index) if "2.0" in list(i)]
    b3_idx = [i for i in list(df.index) if "3.0" in list(i)]
    #print(b3_idx)
    #print(large_rew_idx)

    plt.plot(np.arange(81), df.loc[b1_idx].PC1, c="royalblue", label="Block 1")
    plt.plot(np.arange(81), df.loc[b2_idx].PC1, c="indianred", label="Block 2")
    plt.plot(np.arange(81), df.loc[b3_idx].PC1, c="mediumseagreen", label="Block 3")


    plt.legend()
    plt.xlabel(f"Time Relative to Choice (s)")
    plt.ylabel(f"PC1 - {per_var[0]}%")

    plt.show()


    
if __name__ == "__main__":
    main()