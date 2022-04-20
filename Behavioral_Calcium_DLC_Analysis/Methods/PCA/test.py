import numpy as np
import pandas as pd
import itertools

from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame(np.arange(32).reshape((4,8)), 
            index = pd.date_range('2016-01-01', periods=4),
            columns=['male ; 0', 'male ; 1','male ; 2','male ; 4','female ; 0','female ; 1','female ; 2','female ; 3',])
print(df.head())

sex = ['Male', 'Female']
age = [0,1,2,3]
df.columns = pd.MultiIndex.from_product([sex, age], names=['Sex', 'Age'])

print(df.head())

pal = sns.color_palette("husl",9)

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

plt.show()
