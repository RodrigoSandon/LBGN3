import matplotlib.pyplot as plt

data = {"(1.0, 'Large', False)": {'+ stable': 13, '+ unstable': 10, '- stable': 10, '- unstable': 6, 'N stable': 6, 'N unstable': 12}, 
        "(1.0, 'Small', False)": {'+ stable': 0, '+ unstable': 0, '- stable': 0, '- unstable': 0, 'N stable': 57, 'N unstable': 0}, 
        "(2.0, 'Large', False)": {'+ stable': 3, '+ unstable': 19, '- stable': 4, '- unstable': 14, 'N stable': 8, 'N unstable': 9}, 
        "(2.0, 'Small', False)": {'+ stable': 0, '+ unstable': 0, '- stable': 0, '- unstable': 0, 'N stable': 53, 'N unstable': 4}, 
        "(3.0, 'Large', False)": {'+ stable': 0, '+ unstable': 19, '- stable': 0, '- unstable': 18, 'N stable': 20, 'N unstable': 0}, 
        "(3.0, 'Small', False)": {'+ stable': 0, '+ unstable': 0, '- stable': 0, '- unstable': 0, 'N stable': 34, 'N unstable': 23}}

x = list(data.keys())

fig, ax = plt.subplots()

# Create a list to store the bottom values for each bar
bottoms = [0] * len(x)

for i, (key, val) in enumerate(data.items()):
    ax.bar(x[i], val['+ stable'], label='+ stable', color='lightgreen')
    bottoms[i] += val['+ stable']
    ax.bar(x[i], val['+ unstable'], bottom=bottoms[i], label='+ unstable', color='green')
    bottoms[i] += val['+ unstable']
    ax.bar(x[i], val['- stable'], bottom=bottoms[i], label='- stable', color='lightcoral')
    bottoms[i] += val['- stable']
    ax.bar(x[i], val['- unstable'], bottom=bottoms[i], label='- unstable', color='red')
    bottoms[i] += val['- unstable']
    ax.bar(x[i], val['N stable'], bottom=bottoms[i], label='N stable', color='lightblue')
    bottoms[i] += val['N stable']
    ax.bar(x[i], val['N unstable'], bottom=bottoms[i], label='N unstable', color='blue')

ax.legend()
plt.show()