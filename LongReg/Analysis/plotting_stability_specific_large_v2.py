import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

# Set Seaborn style
sns.set_style("whitegrid")

data = {"(1.0, 'Large', False)": {'+ stable': 13, '+ unstable': 10, '- stable': 10, '- unstable': 6, 'N stable': 6, 'N unstable': 12}, 
        "(2.0, 'Large', False)": {'+ stable': 3, '+ unstable': 19, '- stable': 4, '- unstable': 14, 'N stable': 8, 'N unstable': 9}, 
        "(3.0, 'Large', False)": {'+ stable': 0, '+ unstable': 19, '- stable': 0, '- unstable': 18, 'N stable': 20, 'N unstable': 0}}

x = list(data.keys())

fig, ax = plt.subplots()

x_labels = ['1', '2', '3']
# Create a list to store the bottom values for each bar
bottoms = [0] * len(x)

# Updated colors
colors = {
    '+ stable': '#ff8378',
    '+ unstable': '#cc241d',
    '- stable': '#78ffff',
    '- unstable': '#076678',
    'N stable': '#a9a9a9',
    'N unstable': '#696969'
}

for i, (key, val) in enumerate(data.items()):
    for category, color in colors.items():
        ax.bar(x[i], val[category], bottom=bottoms[i], label=category, color=color)
        bottoms[i] += val[category]

# Get the legend handles and labels
h, l = ax.get_legend_handles_labels()

# Set the legend
#ax.legend(h[:6], l[:6], loc='upper center', bbox_to_anchor=(0.5, -0.075),
          #fancybox=True, shadow=True, ncol=6)

ax.set_xticklabels(x_labels, fontsize=20, fontweight='bold')
ax.tick_params(axis='both', labelsize=20)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontproperties(font_manager.FontProperties(size=20, weight='bold'))


#ax.set_ylabel("Number of Cells")
#ax.set_xlabel("Block")
#ax.set_title("Large Reward + / - / N (No Shock) Cells Stability as a Function of Punishment Probability")
fig.set_size_inches(9, 9)
fig.savefig("/media/rory/Padlock_DT/Prezis/012723/plotting_stability_specific_large.png")
