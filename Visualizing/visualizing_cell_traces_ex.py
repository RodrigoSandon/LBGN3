import os, glob
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Example Data
t_pos = np.arange(0.0, 10.0, 0.1)
t_neg = np.arange(-10.0, 0.0, 0.1)
t = t_neg.tolist() + t_pos.tolist()
t = [round(i, 1) for i in t]

#Block_Reward_Size_Choice_Time_s_1dot0_Large
subevent = "(1.0, 'Large')"
cell_traces_csv = f"/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/RDT D1/Block_Reward Size_Choice Time (s)/{subevent}/all_concat_cells_z_fullwindow.csv"
df = pd.read_csv(cell_traces_csv)
# Only getting first 10 cells for now
df = df.iloc[:, :15]
#print(df.head())
print(len(list(df.columns)))

for count, col in enumerate(list(df.columns)):
    if count > 0:
        for idx in range(len(list(df.index))):
            df.at[idx, col] = df.at[idx, col] + count*4

print(len(list(df.columns)))
# multiple line plot
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_title(subevent)
ax.set_yticks([])
ax.set_ylabel("Neuron #")
ax.get_xaxis().set_visible(True) # doesn't override the off selection of axis

for col in list(df.columns):
    ax.plot(t, list(df[col]), marker='', color='black', linewidth=2)

#plt.legend()
plt.show()