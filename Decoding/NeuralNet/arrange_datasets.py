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

# created per each subevent
class TrialSet:
    def __init__(self, train_prop, val_prop, test_prop):
        self.csv_trial_set = []

        self.num_trials = 0
        self.train_prop = train_prop
        self.val_prop = val_prop
        self.test_prop = test_prop

        self.train_num = 0
        self.val_num = 0
        self.test_num = 0

        self.train_trial_set = []
        self.val_trial_set = []
        self.test_trial_set = []

    def trim(self, amount_to_keep):
        # trim all trial_sets except the limiting one
        # if subtracting amount to keep from len csv_trial_set is zero, then this trial_set is the lim factor
        number_to_delete = len(self.csv_trial_set) - amount_to_keep
        if number_to_delete == 0:
            print("Did not trim!")
        else:
            # this will replace curr csv_trial_set
            self.num_trials = amount_to_keep
            self.csv_trial_set = random.sample(self.csv_trial_set, amount_to_keep)

    def calculate_dataset_proportions(self):
        self.train_num = math.trunc(self.num_trials * self.train_prop)
        self.val_num = math.trunc(self.num_trials * self.val_prop)
        self.test_num = self.num_trials - (self.train_num + self.val_num)

    def check(self) -> None:
        if self.csv_trial_set:
            print(f"List is not empty: {len(self.csv_trial_set)}")
        else:
            print("List is empty!")

    def load_train(self):
        self.train_trial_set = random.sample(self.csv_trial_set, self.train_num)
        for i in self.train_trial_set:
            self.csv_trial_set.remove(i)

    def load_test(self):
        self.test_trial_set = random.sample(self.csv_trial_set, self.test_num)
        for i in self.test_trial_set:
            self.csv_trial_set.remove(i)

    def load_val(self):
        self.val_trial_set = random.sample(self.csv_trial_set, self.val_num)
        for i in self.val_trial_set:
            self.csv_trial_set.remove(i)

    def add_csv(self, file):
        self.csv_trial_set.append(file)
        self.num_trials += 1
        return self


def main():
    ROOT = "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Pearson_Correlation_Datasets_rew/"
    DST_ROOT = "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Pearson_Input_Datasets/Neural_Net_2"
    # ex input: /media/rory/Padlock_DT/BLA_Analysis/Decoding/Pearson_Correlation_Datatrial_sets/BLA-Insc-1/RDT D1/Shock Ocurred_Choice Time (s)/True/trial_1_corrmap.csv
    # ex output: /media/rory/Padlock_DT/BLA_Analysis/Decoding/Pearson_Input_Datatrial_sets/Neural_Net/BLA-Insc-1/RDT D1/Shock Ocurred_Choice Time (s)/train/True/trial_1_corrmap.csv
    
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

    tracker_d = {}

    # get the minimum subtrial_set number
    for session, event in d.items():
        tracker_d[session] = {}
        min_num_trials = 999999
        for event, subevent in d[session].items():
            tracker_d[session][event] = {}
            for subevent, trial_set in d[session][event].items():
                # in the Trialtrial_set now
                trial_set: TrialSet
                if trial_set.num_trials < min_num_trials:
                    min_num_trials = trial_set.num_trials
                
                # the sample of csv paths randomly chosen for train/val/test are not
                # repeats because we remove those samples that were randomly chosen
                # right after loading it onto the train/val/test_trial_set (list)
        for event, subevent in d[session].items():
            tracker_d[session][event] = {}
            for subevent, trial_set in d[session][event].items():
                tracker_d[session][event][subevent] = min_num_trials
                #print(session, event, subevent, trial_set.num_trials)
    #print()
    # trim, now we have 
    for session, event in d.items():
        for event, subevent in d[session].items():
            for subevent, trial_set in d[session][event].items():
                # in the Trialtrial_set now
                trial_set: TrialSet
                #print(tracker_d[session][event][subevent])
                trial_set.trim(tracker_d[session][event][subevent])
                #print(session, event, subevent, trial_set.num_trials)
    # load train/val/test
    for session, event in d.items():
        for event, subevent in d[session].items():
            for subevent, trial_set in d[session][event].items():
                # in the Trialtrial_set now
                trial_set: TrialSet
                print(session, event, subevent, trial_set.num_trials)

                trial_set.calculate_dataset_proportions()
                #print(f"Train: {trial_set.train_num} | Validation: {trial_set.val_num} | Test: {trial_set.test_num}")

                trial_set.load_train()
                trial_set.load_val()
                trial_set.load_test()
                trial_set.check()
                print(f"Train: {len(trial_set.train_trial_set)} | Validation: {len(trial_set.val_trial_set)} | Test: {len(trial_set.test_trial_set)}")
                #print(*trial_set.val_trial_set, sep="\n")
        
    ##### NOW COPY/PASTE THOSE RANDOMLY CHOSEN CSV FILES TO APPROPRIATE DIRS #####
    # Reminder: each session type is it's own decoding session
    for session, event in d.items():
        for event, subevent in d[session].items():
            for subevent, trial_set in d[session][event].items():
                new_dir_train = os.path.join(DST_ROOT, session, event, "train", subevent)
                new_dir_test = os.path.join(DST_ROOT, session, event, "test", subevent)
                new_dir_val = os.path.join(DST_ROOT, session, event, "val", subevent)

                os.makedirs(new_dir_train, exist_ok=True)
                os.makedirs(new_dir_test, exist_ok=True)
                os.makedirs(new_dir_val, exist_ok=True)
                
                """print(session, event, subevent, trial_set.num_trials)
                print("INTERSECIONS")
                
                print(any(i in trial_set.train_trial_set for i in trial_set.val_trial_set))
                print(any(i in trial_set.train_trial_set for i in trial_set.test_trial_set))
                print(any(i in trial_set.test_trial_set for i in trial_set.val_trial_set))
"""
                print("TRAIN trial_set")
                for csv in trial_set.train_trial_set:
                    csv_path = Path(csv)
                    mouse = csv_path.parts[7]
                    filename = csv_path.name
                    new_path = os.path.join(new_dir_train, filename)
                    #print(f"source: {csv1} | dest: {new_path1}")
                    # add mouse name to file, else we get lower file counts
                    new_path = new_path.replace(".csv", f"_{mouse}.csv")
                    shutil.copy(csv, new_path)
                #print(len(trial_set.train_trial_set))

                print("VAL trial_set")
                for csv in trial_set.val_trial_set:
                    csv_path = Path(csv)
                    mouse = csv_path.parts[7]
                    filename = csv_path.name
                    new_path = os.path.join(new_dir_val, filename)
                    #print(f"source: {csv1} | dest: {new_path1}")
                    # add mouse name to file, else we get lower file counts
                    new_path = new_path.replace(".csv", f"_{mouse}.csv")
                    shutil.copy(csv, new_path)
                #print(len(trial_set.val_trial_set))
                
                print("TEST trial_set")
                for csv in trial_set.test_trial_set:
                    csv_path = Path(csv)
                    mouse = csv_path.parts[7]
                    filename = csv_path.name
                    new_path = os.path.join(new_dir_test, filename)
                    #print(f"source: {csv1} | dest: {new_path1}")
                    # add mouse name to file, else we get lower file counts
                    new_path = new_path.replace(".csv", f"_{mouse}.csv")
                    shutil.copy(csv, new_path)
                #print(len(trial_set.test_trial_set))
    
    # Option 1: Combine all trial corrmaps of specific subevent for training and val, test for any corrmap of any mouse (most data, most generalizable)
        # would work well if the model is able to account for multiple microcircuit states that describe one class
    # Option 2: Combine all trial corrmaps of specific subevent for training and val, test within each mouse (test datatrial_set limited to how many trials there are left to test for that mouse)
    # Option 3: Do train/val/test within each mouse's datatrial_set (can't run on some mice)

    
if __name__ == "__main__":
    main()


