import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# Set Seaborn style
sns.set_style("whitegrid")

data = {"(1.0, 'Small', False)": {'+ stable': 0, '+ unstable': 0, '- stable': 0, '- unstable': 0, 'N stable': 57, 'N unstable': 0},  
        "(2.0, 'Small', False)": {'+ stable': 0, '+ unstable': 0, '- stable': 0, '- unstable': 0, 'N stable': 53, 'N unstable': 4}, 
        "(3.0, 'Small', False)": {'+ stable': 0, '+ unstable': 0, '- stable': 0, '- unstable': 0, 'N stable': 34, 'N unstable': 23}}

x = list(data.keys())

# Customize x-axis labels
x_labels = ['1', '2', '3']

fig, ax = plt.subplots()

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

# Set larger font size and bold font for y-axis and x-axis labels
#ax.set_ylabel("Number of Cells", fontsize=14, fontweight='bold')
#ax.set_xlabel("Block", fontsize=14, fontweight='bold')
ax.set_xticklabels(x_labels, fontsize=20, fontweight='bold')
ax.tick_params(axis='both', labelsize=20)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontproperties(font_manager.FontProperties(size=20, weight='bold'))

#ax.set_title("Small Reward + / - / N Cells Stability as a Function of Punishment Probability", fontsize=14, fontweight='bold')
fig.set_size_inches(9, 9)
fig.savefig("/media/rory/Padlock_DT/Prezis/012723/plotting_stability_specific_small.png")