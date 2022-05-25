import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import pandas as pd
from statistics import mean
from operator import attrgetter
import seaborn as sns
from scipy import stats

event = "Block_Reward Size_Choice Time (s)"
rew_type = "Small"
name_of_png = f"all_blocks_{event}_{rew_type}_corr_coef.png"

plt.plot(["1", "2", "3"],[0.120, -0.001, -0.009],'-^k',markersize=10)
plt.ylim((-0.5, 0.5))
plt.xlabel("Block")
plt.ylabel("Correlation Coefficient")
plt.title("Correlation Between Speed & Neural Activity Across Large Reward Blocks")

# to have proper time period, just insert the -10 to 10 into x axis 
plt.savefig(f"/media/rory/Padlock_DT/BLA_Analysis/Results/{name_of_png}")