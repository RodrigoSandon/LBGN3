import pandas as pd

list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7, 8]

replace_func = lambda x: "found" if x in list2 else x
list1 = pd.Series(list1).apply(replace_func).tolist()

print(list1)