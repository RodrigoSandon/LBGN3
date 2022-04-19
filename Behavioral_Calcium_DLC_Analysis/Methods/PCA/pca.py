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
    print(list(df.columns))

    # Run PCA for each of the timepoints (the z scores are averaged across trials)
    #print(subset_df.head())
    pca_obj = PCA()

    pca_obj.fit(df)
    pca_data = pca_obj.transform(df)
    per_var = np.round(pca_obj.explained_variance_ratio_*100, decimals=1)
    labels = ['PC' + str(x) for x in range(1,len(per_var)+1)]
    neurons = list(df.columns)

    pca_df = pd.DataFrame(pca_data, index=neurons, columns = labels)

    return pca_df, per_var

def main():
    blocks = ["1.0", "2.0", "3.0"]

    for block in blocks:
        csv_L = f"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Arranged_Dataset/{block}/Large/BLA-Insc-1/RDT D1/trials_average_corrmap.csv"
        
        pca_df_L, per_var_L = pca_df(csv_L)

        csv_S = f"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Arranged_Dataset/{block}/Small/BLA-Insc-1/RDT D1/trials_average_corrmap.csv"
        
        pca_df_S, per_var_S = pca_df(csv_S)

        plt.scatter(pca_df_L.PC1, pca_df_L.PC2, c=['#1f77b4'])
        plt.scatter(pca_df_S.PC1, pca_df_S.PC2, c=['#ff7f0e'])
        plt.title("PCA Graph")

        plt.xlabel(f"PC1 (L)- {per_var_L[0]}% | PC1 (S)- {per_var_S[0]}%")
        plt.ylabel(f"PC2 (L)- {per_var_L[1]}% | PC2 (S)- {per_var_S[1]}%")

        """for sample in pca_df_L.index:
            #print(type(pca_df.PC1.loc[sample]))
            plt.annotate(sample, (pca_df_L.PC1.loc[sample], pca_df_L.PC2.loc[sample]))
            plt.annotate(sample, (pca_df_S.PC1.loc[sample], pca_df_S.PC2.loc[sample]))"""

        plt.show()


    
if __name__ == "__main__":
    main()