import os, glob
import shutil
import random
import math
from pathlib import Path
from typing import List, Dict

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
        self.csv_set = []

        self.num_trials = 0
        self.train_prop = train_prop
        self.val_prop = val_prop
        self.test_prop = test_prop

        self.train_num = 0
        self.val_num = 0
        self.test_num = 0

        self.train_set = []
        self.val_set = []
        self.test_set = []
    
    def calculate_dataset_proportions(self):
        self.train_num = math.trunc(self.num_trials * self.train_prop)
        self.val_num = math.trunc(self.num_trials * self.val_prop)
        self.test_num = self.num_trials - (self.train_num + self.val_num)

    def check(self) -> None:
        if self.csv_set:
            print(f"List is not empty: {len(self.csv_set)}")
        else:
            print("List is empty!")

    def load_train(self):
        self.train_set = random.sample(self.csv_set, self.train_num)
        for i in self.train_set:
            self.csv_set.remove(i)

    def load_test(self):
        self.test_set = random.sample(self.csv_set, self.test_num)
        for i in self.test_set:
            self.csv_set.remove(i)

    def load_val(self):
        self.val_set = random.sample(self.csv_set, self.val_num)
        for i in self.val_set:
            self.csv_set.remove(i)

    def add_csv(self, file):
        self.csv_set.append(file)
        self.num_trials += 1
        return self


def main():
    ROOT = "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Pearson_Correlation_Datasets/"
    DST_ROOT = "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Pearson_Input_Datasets/Neural_Net"
    # ex input: /media/rory/Padlock_DT/BLA_Analysis/Decoding/Pearson_Correlation_Datasets/BLA-Insc-1/RDT D1/Shock Ocurred_Choice Time (s)/True/trial_1_corrmap.csv
    # ex output: /media/rory/Padlock_DT/BLA_Analysis/Decoding/Pearson_Input_Datasets/Neural_Net/BLA-Insc-1/RDT D1/Shock Ocurred_Choice Time (s)/train/True/trial_1_corrmap.csv
    
    # session: event: subevent: number_of_trials
    d: dict[str, dict[str, dict[str, TrialSet]]] = {}

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
    train_prop = 0.7
    val_prop = 0.1
    test_prop = 0.2

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
                    d[session][event][subevent].add_csv(f)
                else:
                    trial_set = TrialSet(train_prop, val_prop, test_prop)
                    d[session][event][subevent] = trial_set.add_csv(f)
            else:
                d[session][event] = {}
        else:
            d[session] = {}

    for session, event in d.items():
        for event, subevent in d[session].items():
            for subevent, set in d[session][event].items():
                # in the TrialSet now
                set: TrialSet
                print(session, event, subevent, set.num_trials)

                set.calculate_dataset_proportions()
                print(f"Train: {set.train_num} | Validation: {set.val_num} | Test: {set.test_num}")

                set.load_train()
                set.load_val()
                set.load_test()
                set.check()
                # the sample of csv paths randomly chosen for train/val/test are not
                # repeats because we remove those samples that were randomly chosen
                # right after loading it onto the train/val/test_set (list)
    
    ##### NOW COPY/PASTE THOSE RANDOMLY CHOSEN CSV FILES TO APPROPRIATE DIRS #####
    # Reminder: each session type is it's own decoding session
    

    
    
    # Option 1: Combine all trial corrmaps of specific subevent for training and val, test for any corrmap of any mouse (most data, most generalizable)
        # would work well if the model is able to account for multiple microcircuit states that describe one class
    # Option 2: Combine all trial corrmaps of specific subevent for training and val, test within each mouse (test dataset limited to how many trials there are left to test for that mouse)
    # Option 3: Do train/val/test within each mouse's dataset (can't run on some mice)

    
if __name__ == "__main__":
    main()


