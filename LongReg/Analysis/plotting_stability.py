import matplotlib.pyplot as plt

data = {"(1.0, 'Large', False)": {'stable': 29, 'unstable': 28},  
        "(2.0, 'Large', False)": {'stable': 15, 'unstable': 42},  
        "(3.0, 'Large', False)": {'stable': 20, 'unstable': 37}, }

x = list(data.keys())
stables = [val['stable'] for val in data.values()]
unstables = [val['unstable'] for val in data.values()]

fig, ax = plt.subplots()
ax.bar(x, stables, label='stable', color = "#b8d2f2")

rects = ax.patches

# Make some labels.
labels = [f"{i}" for i in stables]

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(
        rect.get_x() + rect.get_width() / 2, height - 5, label, ha="center", va="bottom"
    )

ax.bar(x, unstables, bottom=stables, label='unstable', color = "#F79D9D")
rects = ax.patches

# Make some labels.
labels = [f"{i}" for i in unstables]

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(
        rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom"
    )

ax.set_xticklabels(x, fontsize=7)
ax.set_yticks(range(0, max(stables + unstables) + 20, (max(stables + unstables) + 1) // 10))
ax.set_xlabel("Subevent")
ax.set_ylabel("Number of Cells")
ax.set_title("Large Reward Responsive (No Shock) Cells Stability as a Function of Punishment Probability")
fig.set_size_inches(8, 6)
ax.legend()
fig.savefig("/media/rory/Padlock_DT/Prezis/012723/plotting_stability_large_rew.png")

data = { "(1.0, 'Small', False)": {'stable': 57, 'unstable': 0}, 
        "(2.0, 'Small', False)": {'stable': 53, 'unstable': 4}, 
        "(3.0, 'Small', False)": {'stable': 34, 'unstable': 23}}

x = list(data.keys())
stables = [val['stable'] for val in data.values()]
unstables = [val['unstable'] for val in data.values()]

fig, ax = plt.subplots()
ax.bar(x, stables, label='stable', color = "#b8d2f2")

rects = ax.patches

# Make some labels.
labels = [f"{i}" for i in stables]

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(
        rect.get_x() + rect.get_width() / 2, height - 5, label, ha="center", va="bottom"
    )

rects = ax.patches

ax.bar(x, unstables, bottom=stables, label='unstable', color = "#F79D9D")

# Make some labels.
labels = [f"{i}" for i in unstables]

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(
        rect.get_x() + rect.get_width() / 2, height + 2, label, ha="center", va="bottom"
    )

ax.set_xticklabels(x, fontsize=7)
ax.set_yticks(range(0, max(stables + unstables) + 10, (max(stables + unstables) + 1) // 10))
ax.set_xlabel("Subevent")
ax.set_ylabel("Number of Cells")
ax.set_title("Small Reward Responsive Cells Stability as a Function of Punishment Probability")
fig.set_size_inches(8, 6)
ax.legend()
fig.savefig("/media/rory/Padlock_DT/Prezis/012723/plotting_stability_small_rew.png")