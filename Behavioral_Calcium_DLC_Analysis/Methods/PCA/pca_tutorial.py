import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

csv = "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Pearson_Correlation_Datasets/BLA-Insc-1/RDT D1/Shock Ocurred_Choice Time (s)/True/trial_1_corrmap.csv"
df = pd.read_csv(csv)
df = df.iloc[:, 1:]
print(df.shape)
#print(df.head())
print(list(df.columns))

pca_obj = PCA()

pca_obj.fit(df)
pca_data = pca_obj.transform(df)
per_var = np.round(pca_obj.explained_variance_ratio_*100, decimals=1)
labels = ['PC' + str(x) for x in range(1,len(per_var)+1)]

plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

# Drawing PCA plot - first put new coordinates created by pca_obj.transform(df) where rows have sample labels and columns have PC labels
neurons = list(df.columns)
pca_df = pd.DataFrame(pca_data, index=neurons, columns = labels)
print(pca_df.head())

plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title("PCA Graph")
plt.xlabel("PC1 - {0}%".format(per_var[0]))
plt.ylabel("PC2 - {0}%".format(per_var[1]))

for sample in pca_df.index:
    #print(type(pca_df.PC1.loc[sample]))
    # since we're plotting by each numpy.float, as we go through list of same ttrials but for diff mice,
    # put it into list and then get average and plot that as a numpy float
    # wait, nvm just take the average for that trial of all mice and treat that as if it were from one mouse
    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))

plt.show()
# We can see there's two clusters identified by PCA

# Lastly, let's look at the loading scores for PC1 to determine which genes had the largest influence on separating the two clusters along the x-axis

loading_scores = pd.Series(pca_obj.components_[0], index=neurons)
# Sort loading scores based on their magnitude
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
# Get the loading scores of top ten contributing neurons
top_10_neurons = sorted_loading_scores[0:10].index.values
print(loading_scores[top_10_neurons])