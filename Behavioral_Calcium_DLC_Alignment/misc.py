import os, glob
from pathlib import Path
from typing import List
from itertools import combinations
import pandas as pd


def create_combos(event_name_list_input: List):
    number_items_to_select = list(range(len(event_name_list_input) + 1))
    for i in number_items_to_select:
        to_select = i
        combs = combinations(event_name_list_input, to_select)
        for x in list(combs):
            print("_".join(x))
            print(list(x))


"""
lst = [
    "block",
    "trialtype",
    "rewardsize",
    "shock",
    "omission",
    "winorloss",
    "learning_strat",
]
create_combos(lst)"""


def find_value(csv_path):
    df = pd.read_csv(csv_path)
    val = df.iloc[31, df.columns.get_loc("Choice Time (s)")]
    print(str(val))
    print(type(str(val)))


"""csv_path = "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/Session-20210518-102215_BLA-Insc-6_RDT_D1/2021-05-18-10-26-03_video_BLA-Insc-6_RDT_D1/BLA-INSC-6 05182021_ABET_GPIO_processed.csv"
find_value(csv_path)"""


def print_listdir(root_path):
    for i in os.listdir(root_path):
        print(i)


root_path = r"/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-5/Session-20210510-093930_BLA-Insc-5_RM_D1/SingleCellAlignmentData/C01"
print_listdir(root_path)
