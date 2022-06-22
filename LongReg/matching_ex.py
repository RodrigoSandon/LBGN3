import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def change_cell_name_back_to_old(cell_name: str, df: pd.DataFrame) -> str:

    new_name_row_idx = list(df[df["New Names"] == cell_name].index.values)[0]
    old_cell_name = df.iloc[new_name_row_idx, df.columns.get_loc("Old Names")]

    return old_cell_name

cell_name = "C09"
record_csv = "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/naming_change_record.csv"
df = pd.read_csv(record_csv)

new_name_row_idx = list(df[df["New Names"] == cell_name].index.values)[0]
old_cell_name = df.iloc[new_name_row_idx, df.columns.get_loc("Old Names")]
print(old_cell_name)

#change_cell_name_back_to_old(cell_name, record_csv)