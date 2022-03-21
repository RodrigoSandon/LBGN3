from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

csv = "/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/RDT D1/Shock Ocurred_Choice Time (s)/True/all_concat_cellsbaseline-10_0_gauss1.5.csv"
data = pd.read_csv(csv)

df = data.unstack().reset_index()
df.columns=["X","Y","Z"]

df["X"]=pd.Categorical(df["X"])
df["X"]=df["X"].cat.codes

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot_trisurf(df["Y"],df["X"], df["Z"], cmap=plt.cm.jet, linewidth=0.01)
ax.set_xlabel("Time(s)", fontweight="bold")
ax.set_ylabel("Cell",fontweight='bold')
ax.set_zlabel("Average Df/f Across Trials", fontweight='bold')
plt.show()