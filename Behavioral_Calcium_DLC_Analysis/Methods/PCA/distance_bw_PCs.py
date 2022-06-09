import os, glob
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import math

def is_same_vector_dim(p, q) -> bool:
    if len(p) == len(q):
        return True
    else:
        return False


def vectors_dim(p, q):
    length = None
    if is_same_vector_dim(p, q) == True:
        length = len(p)
    else:
        pass

    return length


def squared_dist(p_ith, q_ith) -> float:
    return ((q_ith - p_ith)**2)


def euclid_dist(p, q):
    n = vectors_dim(p, q)
    sum = 0

    try:
        res = []
        for i in range(n):
            sum += squared_dist(p[i], q[i])
            sqrt = math.sqrt(sum)
            res.append(sqrt)
        return res
    except TypeError:
        print(f"Vectors are not of same dimensions!")

def distance(p, q):
    length = len(p)

    dists = []
    for i in range(length):
        dist = p[i] - q[i]
        dists.append(abs(dist))
    return dists


def euclid_dist_alex(t1, t2):
    return math.sqrt(sum((t1-t2)**2))

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
        "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Unnorm_3Bin_Generalized_PCA_-3_5/2.0/Large/False/RDT D1/all_cells_avg_trials.csv",
        "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Unnorm_3Bin_Generalized_PCA_-3_5/3.0/Large/False/RDT D1/all_cells_avg_trials.csv"
        ]

    t_pos = np.arange(0.0, 5.1, 0.1)
    t_neg = np.arange(-3.0, 0.0, 0.1)
    t = t_neg.tolist() + t_pos.tolist()
    t = [round(i, 1) for i in t]
    
    concatenated_df = combiner_same_block(csvs_to_concat)

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
    
    df, per_var, labels = pca_df(concatenated_df)

    print("PCA DF")
    print(df.head())
    per_var = per_var[0:10]

    b1_idx = [i for i in list(df.index) if "1.0" in list(i)]
    b2_idx = [i for i in list(df.index) if "2.0" in list(i)]
    b3_idx = [i for i in list(df.index) if "3.0" in list(i)]


    res_1 = distance(df.loc[b1_idx].PC1, df.loc[b2_idx].PC1)
    res_2 = distance(df.loc[b1_idx].PC1, df.loc[b3_idx].PC1)
    res_3 = distance(df.loc[b2_idx].PC1, df.loc[b3_idx].PC1)
    plt.plot(t, res_1, c="#9D33FF", label="B1 & B2")
    plt.plot(t, res_2, c="#33FFB0", label="B1 & B3")
    plt.plot(t, res_3, c="#FFD633", label="B2 & B3")
    print(f"Euclidean distance = {res_1}")
    """plt.plot(t, df.loc[b1_idx].PC1, c="royalblue", label="Block 1")
    plt.plot(t, df.loc[b2_idx].PC1, c="indianred", label="Block 2")
    plt.plot(t, df.loc[b3_idx].PC1, c="mediumseagreen", label="Block 3")"""


    plt.legend()
    plt.xlabel(f"Time Relative to Choice (s)")
    plt.ylabel(f"PC1 - {per_var[0]}% Difference")

    plt.show()


    
if __name__ == "__main__":
    main()