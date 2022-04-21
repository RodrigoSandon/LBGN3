import os, glob
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

from matplotlib import animation 
from IPython.display import HTML

def find_paths_endswith(root_path, endswith) -> list:

    files = glob.glob(
        os.path.join(root_path, "*%s") % (endswith), recursive=True,
    )

    return files

def pca_df(csv) :
    df = pd.read_csv(csv)
    df = df.iloc[:, 1:]
    df = df.T # change
    #print(df.head())
    #print(list(df.index))
    arr = df.to_numpy()
    #print(arr)
    #print(arr.shape)

    # Run PCA for each of the timepoints (the z scores are averaged across trials)
    #print(subset_df.head())

    # wee need there to be max 170 principal componentts, need number of labels to be 170
    #when apply no transformation, we get 31*31, when we apply tranformation before, we get 170*31 with
    # there being 31 pc's, so data transformed/untransformed leads to same results of number of PC's

    pca_obj = PCA()

    pca_obj.fit(arr) #here we calculate loading scores and variation each PC accounts for
    #print(pca_obj.explained_variance_ratio_)
    pca_data = pca_obj.transform(arr) #this is where we generate coordinates for pca graph based on loadiing scores
    #print(pca_data.shape)
    per_var = np.round(pca_obj.explained_variance_ratio_*100, decimals=1) #percent variation each pc accounts for
    labels = ['PC' + str(x) for x in range(1,len(pca_data)+1)]
    print(len(labels))
    #CHANGE TO NEURONS IF CORRMAPS
    #neurons = list(df.columns)
    #neurons = list(df.index) #CHANGE TO NEURONS IF CORRMAPS
    time = list(df.columns)

    pca_df = pd.DataFrame(pca_data.T, index=time, columns = labels) #change
    print(pca_df.head())
    #print(pca_df.shape)

    return pca_df, per_var

def main():
    blocks = ["1.0", "2.0", "3.0"]

    #for block in blocks:
        
    csv_L_1 = f"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Generalized_PCA_-3_5/1.0/Large/RDT D1/all_cells_avg_trials.csv"
    
    pca_df_L_1, per_var_L_1 = pca_df(csv_L_1)

    plt.scatter(pca_df_L_1.PC1, pca_df_L_1.PC2, c=['#0000FF'], label="Large, 1")

    csv_L_2 = f"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Generalized_PCA_-3_5/2.0/Large/RDT D1/all_cells_avg_trials.csv"
    
    pca_df_L_2, per_var_L_2 = pca_df(csv_L_2)

    plt.scatter(pca_df_L_2.PC1, pca_df_L_2.PC2, c=['#00FFFF'], label="Large, 2")

    csv_L_3 = f"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Generalized_PCA_-3_5/3.0/Large/RDT D1/all_cells_avg_trials.csv"
    
    pca_df_L_3, per_var_L_3 = pca_df(csv_L_3)

    plt.scatter(pca_df_L_3.PC1, pca_df_L_3.PC2, c=['#069AF3'], label="Large, 3")
    #########################################################################################################

    csv_S_1 = f"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Generalized_PCA_-3_5/1.0/Small/RDT D1/all_cells_avg_trials.csv"
    
    pca_df_S_1, per_var_S_1 = pca_df(csv_S_1)
    
    plt.scatter(pca_df_S_1.PC1, pca_df_S_1.PC2, c=['#DC143C'], label="Small, 1")

    csv_S_2 = f"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Generalized_PCA_-3_5/2.0/Small/RDT D1/all_cells_avg_trials.csv"
    
    pca_df_S_2, per_var_S_2 = pca_df(csv_S_2)
    
    plt.scatter(pca_df_S_2.PC1, pca_df_S_2.PC2, c=['#800000'], label="Small, 2")

    csv_S_3 = f"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Generalized_PCA_-3_5/3.0/Small/RDT D1/all_cells_avg_trials.csv"
    
    pca_df_S_3, per_var_S_3 = pca_df(csv_S_3)
    
    plt.scatter(pca_df_S_3.PC1, pca_df_S_3.PC2, c=['#FF0000'], label="Small, 3")


    plt.legend()
    plt.title("PCA Graph")
    plt.xlabel(f"PC1")
    plt.ylabel(f"PC2")

    """for sample in pca_df_L.index:
        #print(type(pca_df.PC1.loc[sample]))
        plt.annotate(sample, (pca_df_L.PC1.loc[sample], pca_df_L.PC2.loc[sample]))
        plt.annotate(sample, (pca_df_S.PC1.loc[sample], pca_df_S.PC2.loc[sample]))"""

    
    plt.show()


    
if __name__ == "__main__":
    main()