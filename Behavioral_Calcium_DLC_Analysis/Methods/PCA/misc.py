import numpy as np

a = [np.array([]) , np.array([])]

print(a[0])

a = np.append(a[0], ["large", "large"])
print(a[0]) 
print(len(a[0]))