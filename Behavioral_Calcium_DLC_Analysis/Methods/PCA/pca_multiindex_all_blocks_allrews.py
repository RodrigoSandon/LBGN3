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
    multi_cols = [[],[],[]] # type: List[List[str]]
    dfs_to_concat = []
  
    for csv in csvs_to_concat:
        df = pd.read_csv(csv)
        df = df.iloc[:, 1:]
        df = df.T
        dfs_to_concat.append(df)
        rew = csv.split("/")[8]
        block = csv.split("/")[7]
        num_cols = len(list(df.columns))
        for count in range(num_cols):
            multi_cols[0].append(rew)
        for count in range(num_cols):
            multi_cols[1].append(block)
        for col in list(df.columns):
            multi_cols[2].append(col)
    
    multi_cols = [np.asarray(mylist) for mylist in multi_cols]
    result = pd.concat(dfs_to_concat, axis=1)
    result.columns = multi_cols

    return result.T

def main():

    session = "RDT D1"
    shock = "False"
    csvs_to_concat = [
        f"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Unnorm_3Bin_Generalized_PCA_-3_0/1.0/Small/{shock}/{session}/all_cells_avg_trials.csv",
        f"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Unnorm_3Bin_Generalized_PCA_-3_0/2.0/Small/{shock}/{session}/all_cells_avg_trials.csv",
        f"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Unnorm_3Bin_Generalized_PCA_-3_0/3.0/Small/{shock}/{session}/all_cells_avg_trials.csv",
        f"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Unnorm_3Bin_Generalized_PCA_-3_0/1.0/Large/{shock}/{session}/all_cells_avg_trials.csv",
        f"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Unnorm_3Bin_Generalized_PCA_-3_0/2.0/Large/{shock}/{session}/all_cells_avg_trials.csv",
        f"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Unnorm_3Bin_Generalized_PCA_-3_0/3.0/Large/{shock}/{session}/all_cells_avg_trials.csv"
        ]
    # small then large
    colors = ["#F5B7B1", "#E74C3C", "#78281F", "#AED6F1", "#3498DB", "#1B4F72"]

    concatenated_df = combiner_same_block(csvs_to_concat)
    print("CONCATENATED DF BEFORE STANDARDIZATION")
    print(concatenated_df)
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
    
    b1_s_idx = [i for i in list(df.index) if "1.0" in list(i) and "Small" in list(i)]
    b2_s_idx = [i for i in list(df.index) if "2.0" in list(i) and "Small" in list(i)]
    b3_s_idx = [i for i in list(df.index) if "3.0" in list(i) and "Small" in list(i)]

    b1_l_idx = [i for i in list(df.index) if "1.0" in list(i) and "Large" in list(i)]
    b2_l_idx = [i for i in list(df.index) if "2.0" in list(i) and "Large" in list(i)]
    b3_l_idx = [i for i in list(df.index) if "3.0" in list(i) and "Large" in list(i)]
    

    fig = plt.figure()
    ax = fig.add_subplot(111,projection="3d")

    ax.scatter(df.loc[b1_s_idx].PC1, df.loc[b1_s_idx].PC2,df.loc[b1_s_idx].PC3, c=colors[0], label="S B1")
    ax.scatter(df.loc[b2_s_idx].PC1, df.loc[b2_s_idx].PC2,df.loc[b2_s_idx].PC3, c=colors[1], label="S B2")
    ax.scatter(df.loc[b3_s_idx].PC1, df.loc[b3_s_idx].PC2,df.loc[b3_s_idx].PC3, c=colors[2], label="S B3")

    ax.scatter(df.loc[b1_l_idx].PC1, df.loc[b1_l_idx].PC2,df.loc[b1_l_idx].PC3, c=colors[3], label="L B1")
    ax.scatter(df.loc[b2_l_idx].PC1, df.loc[b2_l_idx].PC2,df.loc[b2_l_idx].PC3, c=colors[4], label="L B2")
    ax.scatter(df.loc[b3_l_idx].PC1, df.loc[b3_l_idx].PC2,df.loc[b3_l_idx].PC3, c=colors[5], label="L B3")

    """xAxisLine = ((min(df.loc[b1_s_idx].PC1), max(df.loc[b1_s_idx].PC1)), (0, 0), (0,0))
    ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
    yAxisLine = ((0, 0), (min(df.loc[b1_s_idx].PC2), max(df.loc[b1_s_idx].PC2)), (0,0))
    ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
    zAxisLine = ((0, 0), (0,0), (min(df.loc[b1_s_idx].PC3), max(df.loc[b1_s_idx].PC3)))
    ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')"""


    ax.legend()
    ax.set_title("PCA Graph")
    ax.set_xlabel(f"PC1 - {per_var[0]}%")
    ax.set_ylabel(f"PC2 - {per_var[1]}%")
    ax.set_zlabel(f"PC3 - {per_var[2]}%")

    
    #plt.show()
    ax.figure.savefig(f"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Results/allblocks_{session}_LARGE_SMALL_{shock}_-3_0_pcaplot.svg")


    
if __name__ == "__main__":
    main()