import pandas as pd
from pandas.core.indexing import check_bool_indexer

"""x: dict = {"yes": [], "no": []}
for count, key in enumerate(x):
    print(count)
    print(key)"""


csv_path = r"/home/rory/Rodrigo/Database/StackedBars/how_shock_respcells_changein_blocks_and_rewsize/take2_rdt1.csv"
session = "RDT_D1"
dst = csv_path.replace(".csv", ".png")

df = pd.read_csv(csv_path)

resp_L_counts: list = []
resp_S_counts: list = []
resp_both_counts: list = []

labels = ["Block 1", "Block 2", "Block 3"]

shock_resp_cells_count = len(df)

# works
for count, col in enumerate(list(df.columns)):
    for idx, cell_id in enumerate(df[col]):
        print(df.loc[idx][col])
    break
# also works
for count, col in enumerate(list(df.columns)):
    for cell_id in df[col]:
        print(cell_id)
    break
