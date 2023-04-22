from cgitb import small
import os, glob
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
from scipy.stats import mannwhitneyu

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
    multi_cols = [[],[], []] # type: List[List[str]]
    dfs_to_concat = []
  
    for csv in csvs_to_concat:
        df = pd.read_csv(csv)
        #print(df.head())
        print(len(df))
        df = df.iloc[:, 1:]
        df = df.T
        #df = df.reset_index(drop=True)  # Reset the index to row numbers
        dfs_to_concat.append(df)
        session = csv.split("/")[9]
        block = csv.split("/")[7]
        num_cols = len(list(df.columns))
        #print(num_cols)
        for col in list(df.columns):
            multi_cols[0].append(session)
        for count in range(num_cols):
            multi_cols[1].append(block)
        for col in list(df.columns):
            multi_cols[2].append(col)
    
    multi_cols = [np.asarray(mylist) for mylist in multi_cols]
    #print(multi_cols)
    result = pd.concat(dfs_to_concat, axis=1)
    #result.to_csv("/media/rory/Padlock_DT/BLA_Analysis/Decoding/test.csv")
    #print(result)
    #print(len(result))
    result.columns = multi_cols

    result = result.dropna()

    return result.T

def combiner_same_block_2():
    pass


def main():

    session = "Pre-RDT RM"
    session_2 = "RDT D1"
    shock = "False"
    starting_timepoint = 70
    ending_timepoint = 151
    dist_dir = f"Unnorm_Generalized_PCA_{starting_timepoint}_{ending_timepoint}"
    base_dir = f"/media/rory/Padlock_DT/BLA_Analysis/Decoding/{dist_dir}"
    outfilename = f"PCA_Results_{session}_{session_2}_shock{shock}_{starting_timepoint}_{ending_timepoint}.png"
    scree_outfilename = f"PCA_Scree_Results_{session}_{session_2}_shock{shock}_{starting_timepoint}_{ending_timepoint}.png"
    out_path = os.path.join(base_dir, outfilename)
    scree_out_path = os.path.join(base_dir, scree_outfilename)

    csvs_to_concat = [
        f"/media/rory/Padlock_DT/BLA_Analysis/Decoding/{dist_dir}/1.0/{shock}/{session}/all_cells_avg_trials.csv",
        f"/media/rory/Padlock_DT/BLA_Analysis/Decoding/{dist_dir}/2.0/{shock}/{session}/all_cells_avg_trials.csv",
        f"/media/rory/Padlock_DT/BLA_Analysis/Decoding/{dist_dir}/3.0/{shock}/{session}/all_cells_avg_trials.csv",
        f"/media/rory/Padlock_DT/BLA_Analysis/Decoding/{dist_dir}/1.0/{shock}/{session_2}/all_cells_avg_trials.csv",
        f"/media/rory/Padlock_DT/BLA_Analysis/Decoding/{dist_dir}/2.0/{shock}/{session_2}/all_cells_avg_trials.csv",
        f"/media/rory/Padlock_DT/BLA_Analysis/Decoding/{dist_dir}/3.0/{shock}/{session_2}/all_cells_avg_trials.csv"
        ]

    # session first, then the second one
    colors = ["#F5B7B1", "#E74C3C", "#78281F", "#AED6F1", "#3498DB", "#1B4F72"]

    t_pos = np.arange(0.0, 5.1, 0.1)
    t_neg = np.arange(-3.0, 0.0, 0.1)
    t = t_neg.tolist() + t_pos.tolist()
    t = [round(i, 1) for i in t]
    
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
    plt.savefig(scree_out_path)
    plt.close()

    b1_rdt_idx = [i for i in list(df.index) if "1.0" in list(i) and session in list(i)]
    #print(list(df.index)[0])
    #print("df.index:", df.index)
    #print("session:", session)
    #print("b1_rdt_idx:", b1_rdt_idx)
    b2_rdt_idx = [i for i in list(df.index) if "2.0" in list(i) and session in list(i)]
    b3_rdt_idx = [i for i in list(df.index) if "3.0" in list(i) and session in list(i)]

    b1_rm_idx = [i for i in list(df.index) if "1.0" in list(i) and session_2 in list(i)]
    b2_rm_idx = [i for i in list(df.index) if "2.0" in list(i) and session_2 in list(i)]
    b3_rm_idx = [i for i in list(df.index) if "3.0" in list(i) and session_2 in list(i)]
    #print(b3_idx)
    #print(large_rew_idx)

    fig = plt.figure()
    ax = fig.add_subplot(111,projection="3d")
   
    #print(len(t),len(df.loc[b1_rdt_idx].PC1) )
    #print(len(t), df.loc[b1_rm_idx].PC1)

    ax.scatter(t, df.loc[b1_rdt_idx].PC1, df.loc[b1_rdt_idx].PC2, c=colors[0], label=f"{session}: B1")
    ax.scatter(t, df.loc[b2_rdt_idx].PC1, df.loc[b2_rdt_idx].PC2, c=colors[1], label=f"{session}: B2")
    ax.scatter(t, df.loc[b3_rdt_idx].PC1, df.loc[b3_rdt_idx].PC2, c=colors[2], label=f"{session}: B3")

    ax.scatter(t, df.loc[b1_rm_idx].PC1, df.loc[b1_rm_idx].PC2, c=colors[3], label=f"{session_2}: B1")
    ax.scatter(t, df.loc[b2_rm_idx].PC1, df.loc[b2_rm_idx].PC2, c=colors[4], label=f"{session_2}: B2")
    ax.scatter(t, df.loc[b3_rm_idx].PC1, df.loc[b3_rm_idx].PC2, c=colors[5], label=f"{session_2}: B3")
    # PLOTTING SIGNIFICANT DIFFERENCES

    # Define a function for multiple comparison correction
    def bonferroni_correction(a_list, p_value, alpha=0.01):
        n = len(a_list)
        print(n)
        corrected_alpha = alpha / n
        print(corrected_alpha)
        return p_value < corrected_alpha

    # Perform pairwise Mann-Whitney U tests and store p-values
    p_values = []

    # Comparisons within the first session
    p_values.append(bonferroni_correction(df.loc[b1_rdt_idx, 'PC1'],mannwhitneyu(df.loc[b1_rdt_idx, 'PC1'], df.loc[b2_rdt_idx, 'PC1']).pvalue))
    p_values.append(bonferroni_correction(df.loc[b1_rdt_idx, 'PC1'], mannwhitneyu(df.loc[b1_rdt_idx, 'PC1'], df.loc[b3_rdt_idx, 'PC1']).pvalue))
    p_values.append(bonferroni_correction(df.loc[b2_rdt_idx, 'PC1'], mannwhitneyu(df.loc[b2_rdt_idx, 'PC1'], df.loc[b3_rdt_idx, 'PC1']).pvalue))

    # Comparisons within the second session
    p_values.append(bonferroni_correction(df.loc[b1_rm_idx, 'PC1'], mannwhitneyu(df.loc[b1_rm_idx, 'PC1'], df.loc[b2_rm_idx, 'PC1']).pvalue))
    p_values.append(bonferroni_correction(df.loc[b1_rm_idx, 'PC1'], mannwhitneyu(df.loc[b1_rm_idx, 'PC1'], df.loc[b3_rm_idx, 'PC1']).pvalue))
    p_values.append(bonferroni_correction(df.loc[b2_rm_idx, 'PC1'], mannwhitneyu(df.loc[b2_rm_idx, 'PC1'], df.loc[b3_rm_idx, 'PC1']).pvalue))

    # Comparisons between sessions
    p_values.append(bonferroni_correction(df.loc[b1_rdt_idx, 'PC1'], mannwhitneyu(df.loc[b1_rdt_idx, 'PC1'], df.loc[b1_rm_idx, 'PC1']).pvalue))
    p_values.append(bonferroni_correction(df.loc[b2_rdt_idx, 'PC1'],mannwhitneyu(df.loc[b2_rdt_idx, 'PC1'], df.loc[b2_rm_idx, 'PC1']).pvalue))
    p_values.append(bonferroni_correction(df.loc[b3_rdt_idx, 'PC1'],mannwhitneyu(df.loc[b3_rdt_idx, 'PC1'], df.loc[b3_rm_idx, 'PC1']).pvalue))

    # Apply Bonferroni correction and print results
    #print(p_values)
    #significant = bonferroni_correction(p_values)

    print("Significant differences:")
    print(f"Session 1: B1 vs B2: {p_values[0]}")
    print(f"Session 1: B1 vs B3: {p_values[1]}")
    print(f"Session 1: B2 vs B3: {p_values[2]}")
    print(f"Session 2: B1 vs B2: {p_values[3]}")
    print(f"Session 2: B1 vs B3: {p_values[4]}")
    print(f"Session 2: B2 vs B3: {p_values[5]}")
    print(f"Between sessions: B1: {p_values[6]}")
    print(f"Between sessions: B2: {p_values[7]}")
    print(f"Between sessions: B3: {p_values[8]}")

    # SHOWING IN THE GRAPH

    """def plot_significance(ax, x1, x2, y1, y2, z1, z2, h_offset=0.2, v_offset=0.5):
        y_upper = max(y1, y2) + v_offset
        
        ax.plot([x1, x1], [y1, y_upper], [z1, z1 + 3], color='gray', linestyle='--', alpha=0.7)
        ax.plot([x2, x2], [y2, y_upper], [z2, z2 + 3], color='gray', linestyle='--', alpha=0.7)
        ax.plot([x1, x2], [y_upper, y_upper], [z1, z2 + 3], color='gray', linestyle='--', alpha=0.7)
        
        x_mid = (x1 + x2) / 2
        z_mid = max(z1 , z2) + 3
        ax.text(x_mid, y_upper + h_offset, z_mid, '*', fontsize=16)




    # Find the time point index for each session at the midpoint
    t1_mid = int(len(df.loc[b1_rdt_idx, 'PC1']) / 2)
    t2_mid = int(len(df.loc[b1_rm_idx, 'PC1']) / 2)

    # For each significant result, plot the significance line
    if significant[0]:
        plot_significance(ax, t[t1_mid], t[t1_mid],
                            df.loc[b1_rdt_idx].PC1.iloc[t1_mid], df.loc[b2_rdt_idx].PC1.iloc[t1_mid],
                            df.loc[b1_rdt_idx].PC2.iloc[t1_mid], df.loc[b2_rdt_idx].PC2.iloc[t1_mid])
    if significant[1]:
        plot_significance(ax, t[t1_mid], t[t1_mid],
                            df.loc[b1_rdt_idx].PC1.iloc[t1_mid], df.loc[b3_rdt_idx].PC1.iloc[t1_mid],
                            df.loc[b1_rdt_idx].PC2.iloc[t1_mid], df.loc[b3_rdt_idx].PC2.iloc[t1_mid])
    if significant[2]:
        plot_significance(ax, t[t1_mid], t[t1_mid],
                            df.loc[b2_rdt_idx].PC1.iloc[t1_mid], df.loc[b3_rdt_idx].PC1.iloc[t1_mid],
                            df.loc[b2_rdt_idx].PC2.iloc[t1_mid], df.loc[b3_rdt_idx].PC2.iloc[t1_mid])
    if significant[3]:
        plot_significance(ax, t[t2_mid], t[t2_mid],
                            df.loc[b1_rm_idx].PC1.iloc[t2_mid], df.loc[b2_rm_idx].PC1.iloc[t2_mid],
                            df.loc[b1_rm_idx].PC2.iloc[t2_mid], df.loc[b2_rm_idx].PC2.iloc[t2_mid])
    if significant[4]:
        plot_significance(ax, t[t2_mid], t[t2_mid],
                            df.loc[b1_rm_idx].PC1.iloc[t2_mid], df.loc[b3_rm_idx].PC1.iloc[t2_mid],
                            df.loc[b1_rm_idx].PC2.iloc[t2_mid], df.loc[b3_rm_idx].PC2.iloc[t2_mid])
    if significant[5]:
        plot_significance(ax, t[t2_mid], t[t2_mid],
                            df.loc[b2_rm_idx].PC1.iloc[t2_mid], df.loc[b3_rm_idx].PC1.iloc[t2_mid],
                            df.loc[b2_rm_idx].PC2.iloc[t2_mid], df.loc[b3_rm_idx].PC2.iloc[t2_mid])
    if significant[6]:
        plot_significance(ax, t[t1_mid], t[t2_mid],
                            df.loc[b1_rdt_idx].PC1.iloc[t1_mid], df.loc[b1_rm_idx].PC1.iloc[t2_mid],
                            df.loc[b1_rdt_idx].PC2.iloc[t1_mid], df.loc[b1_rm_idx].PC2.iloc[t2_mid])
    if significant[7]:
        plot_significance(ax, t[t1_mid], t[t2_mid],
                            df.loc[b2_rdt_idx].PC1.iloc[t1_mid], df.loc[b2_rm_idx].PC1.iloc[t2_mid],
                            df.loc[b2_rdt_idx].PC2.iloc[t1_mid], df.loc[b2_rm_idx].PC2.iloc[t2_mid])
    if significant[8]:
        plot_significance(ax, t[t1_mid], t[t2_mid],
                            df.loc[b3_rdt_idx].PC1.iloc[t1_mid], df.loc[b3_rm_idx].PC1.iloc[t2_mid],
                            df.loc[b3_rdt_idx].PC2.iloc[t1_mid], df.loc[b3_rm_idx].PC2.iloc[t2_mid])
"""
    
    #fig.set_size_inches(12, 10)
    plt.subplots_adjust(right=0.9, top=0.8)
    #plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.3))
    #ax.set_title("PCA Graph")
    ax.set_ylabel(f"PC1 - {per_var[0]}%")
    ax.set_zlabel(f"PC2 - {per_var[1]}%")
    ax.set_xlabel(f"Time relative to choice (s)")
    
    plt.savefig(out_path)


    
if __name__ == "__main__":
    main()