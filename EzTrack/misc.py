import matplotlib.pyplot as plt

x = [1,2,3,4,5,6]
y = [1,2,3,4,5,6]
mylst = [0,0,"first",0,"second",0]

fig, ax = plt.subplots()

# plot the line chart
ax.plot(x, y)

# set the x-axis tick labels
tick_labels = []
for i, label in enumerate(mylst):
    if isinstance(label, str):
        tick_labels.append(label)
    else:
        print(x[i])
        tick_labels.append(str(x[i]))
print(tick_labels)
ax.set_xticklabels(tick_labels)

# set the x-axis label
ax.set_xlabel("Index")

# set the y-axis label
ax.set_ylabel("Value")

# show the plot
plt.show()
plt.close()

x = [1,"first",3,4,5,6]
y = [1,2,3,4,5,6]
mylst = [0,0,"first",0,"second",0]

plt.plot(x, y)
plt.show()