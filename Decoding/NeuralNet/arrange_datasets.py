import os, glob
import shutil
import random
import math
from pathlib import Path
from typing import List

def find_paths(root_path: Path, endswith: str) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, "**", endswith), recursive=True,
    )
    return files

def find_paths_num_paths_no_btwn_per_mouse(root_path: Path, endswith: str) -> int:
    files = glob.glob(
        os.path.join(root_path, endswith), recursive=True,
    )
    return len(files)

def find_paths_num_paths_all_mice(root_path: Path, endswith: str) -> int:
    files = glob.glob(
        os.path.join(root_path,"**", endswith), recursive=True,
    )
    return len(files)

class TrialSet:
    def __init__(self, train_prop, val_prop, test_prop):
        self.num_trials = 0
        self.train_prop = train_prop
        self.val_prop = val_prop
        self.test_prop = test_prop

        self.csv_set = []
    
    def calculate_dataset_proportions(self) -> int:
        train_num = math.trunc(self.num_trials * self.train_prop)
        val_num = math.trunc(self.num_trials * self.val_prop)
        test_num = self.num_trials - (train_num + val_num)
        
        return train_num, val_num, test_num

    def check(self) -> None:
        if self.csv_set:
            print(f"List is not empty: {len(self.csv_set)}")
        else:
            print("List is empty!")

    def load_dataset(self, num) -> list:
        rand_sample = random.sample(self.csv_set, num)
        self.csv_set.remove(rand_sample)
        return rand_sample

    def add_csv(self, file):
        self.csv_set.append(file)
        self.num_trials += 1


def main():
    ROOT = "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Pearson_Correlation_Datasets/"
    DST_ROOT = "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Pearson_Input_Datasets/Neural_Net"
    # ex input: /media/rory/Padlock_DT/BLA_Analysis/Decoding/Pearson_Correlation_Datasets/BLA-Insc-1/RDT D1/Shock Ocurred_Choice Time (s)/True/trial_1_corrmap.csv
    # ex output: /media/rory/Padlock_DT/BLA_Analysis/Decoding/Pearson_Input_Datasets/Neural_Net/BLA-Insc-1/RDT D1/Shock Ocurred_Choice Time (s)/train/True/trial_1_corrmap.csv
    
    # mouse: session: event: subevent: number_of_trials
    d = {}

    files = find_paths(ROOT, f"trial_*_corrmap.csv")
    
    ##### GETTING INDIVIDUAL MOUSE TRIAL COUNTS #####
    """for f in files:
        # need to count how many trials are in each subevent
        # to find out limiting factor
        f = Path(f)

        mouse = f.parts[7]
        session = f.parts[8]
        event = f.parts[9]
        subevent = f.parts[10]

        root = f.parent
        num_trials = find_paths_num_paths_no_btwn(root, "trial_*_corrmap.csv")
        
        if mouse not in d:
            d[mouse] = {}
        elif mouse in d:
            if session not in d[mouse]:
                d[mouse][session] = {}
            elif session in d[mouse]:
                if event not in d[mouse][session]:
                    d[mouse][session][event] = {}
                elif event in d[mouse][session]:
                    d[mouse][session][event][subevent] = num_trials

    for key, value in d.items():
        print(key, value)"""
    ##### GETTING ALL MOUSE TRIAL COUNTS #####
    for f in files:
        # need to count how many trials are in each subevent
        # to find out limiting factor
        f = Path(f)

        session = f.parts[8]
        event = f.parts[9]
        subevent = f.parts[10]

        if session in d:
            if event in d[session]:
                if subevent in d[session][event]:
                    d[session][event][subevent] += 1
                else:
                    d[session][event][subevent] = 1
            else:
                d[session][event] = {}
        else:
            d[session] = {}

    for key, value in d.items():
        print(key, value)
    
    
    # Option 1: Combine all trial corrmaps of specific subevent for training and val, test for any corrmap of any mouse (most data, most generalizable)
        # would work well if the model is able to account for multiple microcircuit states that describe one class
    # Option 2: Combine all trial corrmaps of specific subevent for training and val, test within each mouse (test dataset limited to how many trials there are left to test for that mouse)
    # Option 3: Do train/val/test within each mouse's dataset (can't run on some mice)

    
if __name__ == "__main__":
    main()


