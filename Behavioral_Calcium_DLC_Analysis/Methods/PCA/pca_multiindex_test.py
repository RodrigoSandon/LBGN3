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
    df = df.T
    
    #print(df)

    pca_obj = PCA()

    pca_obj.fit(df)
    
    pca_data = pca_obj.transform(df)
 
    per_var = np.round(pca_obj.explained_variance_ratio_*100, decimals=1)
    labels = ['PC' + str(x) for x in range(1,len(pca_data)+1)]
    print(len(labels))

    time = list(df.columns)

    pca_df = pd.DataFrame(pca_data.T, index=time, columns = labels)
    print(pca_df)
    

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
        rew_size = csv.split("/")[8]
        num_cols = len(list(df.columns))
        for count in range(num_cols):
            multi_cols[0].append(rew_size)
        for col in list(df.columns):
            multi_cols[1].append(col)
    
    multi_cols = [np.asarray(mylist) for mylist in multi_cols]
    result = pd.concat(dfs_to_concat, axis=1)
    result.columns = multi_cols

    return result.T

def main():

    csvs_to_concat = [
        "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Generalized_PCA_-3_5/1.0/Small/RDT D1/all_cells_avg_trials.csv",
        "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Generalized_PCA_-3_5/1.0/Large/RDT D1/all_cells_avg_trials.csv"
        ]
    
    concatenated_df = combiner_same_block(csvs_to_concat)
    #print(concatenated_df)
    #print(len(concatenated_df["Small"].columns))
    #print(len(concatenated_df["Large"].columns))

    df, per_var,labels = pca_df(concatenated_df)
    #print(df)
    per_var = per_var[0:10]
    labels = labels[0:10]
    plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.show()

    #df.index = [list(i) for i in list(df.index)]
    #print(list(df.index))
    #print(type(list(df.index)[0]))

    small_rew_idx = [i for i in list(df.index) if "Small" in list(i)]
    large_rew_idx = [i for i in list(df.index) if "Large" in list(i)]
 
    #fig = plt.figure()
    #ax = fig.add_subplot(111,projection="3d")
   
    plt.scatter(df.loc[small_rew_idx].PC1, df.loc[small_rew_idx].PC2, c="royalblue", label="Small")
    plt.scatter(df.loc[large_rew_idx].PC1, df.loc[large_rew_idx].PC2, c="indianred", label="Large")

    """xAxisLine = ((min(df.loc[small_rew_idx].PC1), max(df.loc[small_rew_idx].PC1)), (0, 0), (0,0))
    ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
    yAxisLine = ((0, 0), (min(df.loc[small_rew_idx].PC2), max(df.loc[small_rew_idx].PC2)), (0,0))
    ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
    zAxisLine = ((0, 0), (0,0), (min(df.loc[small_rew_idx].PC3), max(df.loc[small_rew_idx].PC3)))
    ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')"""

    
    plt.legend()
    plt.title("PCA Graph")
    plt.xlabel(f"PC1 - {per_var[0]}%")
    plt.ylabel(f"PC2 - {per_var[1]}%")
    """
    ax.legend()
    ax.set_title("PCA Graph")
    ax.set_xlabel(f"PC1 - {per_var[0]}%")
    ax.set_ylabel(f"PC2 - {per_var[1]}%")
    ax.set_zlabel(f"PC3 - {per_var[2]}%")"""

    
    plt.show()


    
if __name__ == "__main__":
    main()