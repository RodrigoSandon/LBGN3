import matplotlib.pyplot as plt

data = {"(1.0, 'Small', False)": {'+ stable': 0, '+ unstable': 0, '- stable': 0, '- unstable': 0, 'N stable': 57, 'N unstable': 0},  
        "(2.0, 'Small', False)": {'+ stable': 0, '+ unstable': 0, '- stable': 0, '- unstable': 0, 'N stable': 53, 'N unstable': 4}, 
        "(3.0, 'Small', False)": {'+ stable': 0, '+ unstable': 0, '- stable': 0, '- unstable': 0, 'N stable': 34, 'N unstable': 23}}

x = list(data.keys())

fig, ax = plt.subplots()

# Create a list to store the bottom values for each bar
bottoms = [0] * len(x)

for i, (key, val) in enumerate(data.items()):
    ax.bar(x[i], val['+ stable'], label='+ stable', color='#ff8378')
    bottoms[i] += val['+ stable']
    ax.bar(x[i], val['+ unstable'], bottom=bottoms[i], label='+ unstable', color='#73160d')
    bottoms[i] += val['+ unstable']
    ax.bar(x[i], val['- stable'], bottom=bottoms[i], label='- stable', color='#78ffff')
    bottoms[i] += val['- stable']
    ax.bar(x[i], val['- unstable'], bottom=bottoms[i], label='- unstable', color='#0d6973')
    bottoms[i] += val['- unstable']
    ax.bar(x[i], val['N stable'], bottom=bottoms[i], label='N stable', color='#86ff78')
    bottoms[i] += val['N stable']
    ax.bar(x[i], val['N unstable'], bottom=bottoms[i], label='N unstable', color='#17730d')

h,l = ax.get_legend_handles_labels()
ax.legend(h[:6], l[:6], loc='upper center', bbox_to_anchor=(0.5, -0.075),
          fancybox=True, shadow=True, ncol=6)
ax.set_ylabel("Number of Cells")
ax.set_xlabel("Block")
ax.set_title("Small Reward + / - / N Cells Stability as a Function of Punishment Probability")
fig.set_size_inches(9, 9)
fig.savefig("/media/rory/Padlock_DT/Prezis/012723/plotting_stability_specific_small.png")