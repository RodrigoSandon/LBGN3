import numpy as np
import pandas as pd
import itertools

from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

"""df = pd.DataFrame(np.arange(32).reshape((4,8)), 
            index = pd.date_range('2016-01-01', periods=4),
            columns=['male ; boy1', 'male ; boy2','male ; boy3','male ; boy4','female ; girl1','female ; girl2','female ; girl3','female ; girl4',])
print(df.head())

sex = ['Male', 'Female']
age = [45,23,34,37]
df.columns = pd.MultiIndex.from_product([sex, age], names=['Sex', 'Age'])

print(df.head())"""

# for one mouse
# some can be missing, so if they're missing one cell from a session, omit them from analysis
arr_1 = [
    ["glob_cell_1", "glob_cell_1", "glob_cell_2", "glob_cell_2", "glob_cell_3", "glob_cell_3"],
    ["C01, Pre-RDT RM", "C03, RDT D1", "C02, Pre-RDT RM", "C01, RDT D1","C03, Pre-RDT RM", "C02, RDT D1", ]
]

tups = list(zip(*arr_1))

index = pd.MultiIndex.from_tuples(tups, names=["Global Cells", "Local Cells"])

s = pd.Series(np.random.randn(6), index=index, name= "df/f")

df: pd.DataFrame

df = s.to_frame()

print(s)
print(df.head())

"""pal = sns.color_palette("husl",9)

pca_obj = PCA()

pca_obj.fit(df)
pca_data = pca_obj.transform(df)
per_var = np.round(pca_obj.explained_variance_ratio_*100, decimals=1)
labels = ['PC' + str(x) for x in range(1,len(per_var)+1)]
#CHANGE TO NEURONS IF CORRMAPS
#neurons = list(df.columns)
time = list(df.index)

pca_df = pd.DataFrame(pca_data, index=time, columns = labels)
print(pca_df.head())

for i, sex in enumerate(list(df.columns)):
    plt.scatter(pca_df.PC1, pca_df.PC2, c= pal[i])

plt.legend()
plt.title("PCA Graph")
plt.xlabel(f"PC1")
plt.ylabel(f"PC2")

plt.show()"""
