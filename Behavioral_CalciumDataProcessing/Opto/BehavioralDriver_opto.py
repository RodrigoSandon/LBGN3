import pandas as pd
import numpy as np
import shutil
import os, glob
from pathlib import Path
from typing import List
from BehavioralSession_opto import BehavioralSession
from BehavioralUtilities_opto import BehavioralUtilities


def find_paths_startswith_and_endswith(root_path, startswith, endswith) -> List:

    files = glob.glob(
        os.path.join(root_path, "**", "%s*%s") % (startswith, endswith),
        recursive=True,
    )

    return files

def parse_abet_file(abet_file_path):

    mouse_date = abet_file_path.split("/")[-1]
    mouse = mouse_date.split(" ")[0]
    date = mouse_date.split(" ")[1].replace(".csv","")

    return mouse, date

def move_abet_file(mouse: str, file_name):
    # ex: /media/rory/RDT VIDS/ABET_files_opto/RRD168 01142021.csv
    # dst: /media/rory/RDT VIDS/BORIS/RRD168/RDT OPTO CHOICE 0114/

    ############ CHANGE HERE: SPECIFY WHERE ABET FILES GONNA GO TO ############
    dst_root = r"/media/rory/RDT VIDS/BORIS_merge/BATCH_2"
    ############ CHANGE HERE: SPECIFY WHERE ABET FILES GONNA GO TO ############
    
    if dst_root == "/media/rory/RDT VIDS/BORIS_merge/BATCH_2":
        dst_dir = os.path.join(dst_root, mouse.lower())
    else:
        dst_dir = os.path.join(dst_root, mouse)

    for dir in os.listdir(dst_dir):
        if ("CHOICE" in dir) or ("choice" in dir):
            #print(f"dir: {dir}")
            dst_dir = os.path.join(dst_dir, dir, file_name)
    
            return dst_dir

def main():

    ROOT = r"/media/rory/RDT VIDS/ABET_files_opto/"
    files = find_paths_startswith_and_endswith(ROOT, "RRD", ".csv")

    for count, f in enumerate(files):
        print(f"file {count}/{len(files)}")
        print("CURRENT PATH: ", f)

        try:
            file_name = f.split("/")[-1]

            mouse, date = parse_abet_file(f)
    
            dst_abet_dir = move_abet_file(mouse, file_name)
            shutil.copy(f, dst_abet_dir)
            f = dst_abet_dir

            correction_file = "/media/rory/Padlock_DT/Opto_Speed_Analysis/detecting_light_dark_frames/opto_abet_file_corrections.csv"
            correction_file_df = pd.read_csv(correction_file)
            to_add = None
            
            for count, path in enumerate(list(correction_file_df["vid_path"])):
                if mouse in path and date in path:
                    to_add = list(correction_file_df["ABET_addition_correction_time_(s)"])[count]
            
            #print(f"Adding {to_add} to times")

            ABET_1 = BehavioralSession(
                f,
            )
            ABET_1.preprocess_csv()
            df = ABET_1.get_df()

            times = [time + to_add for time in list(df["Evnt_Time"])]
            df["Evnt_Time"] = times

            #print(df.head())

            grouped_by_trialnum = df.groupby("trial_num")
            processed_behavioral_df = grouped_by_trialnum.apply(
                BehavioralUtilities.process_csv
            )  # is a new df, it's not the modified df

            processed_behavioral_df = (
                BehavioralUtilities.add_winstay_loseshift_loseomit(
                    processed_behavioral_df
                )
            )
            # Add post-hoc processing
            processed_behavioral_df = BehavioralUtilities.shift_col_values(
                processed_behavioral_df
            )
            
            processed_behavioral_df = BehavioralUtilities.interpolate_block(
                processed_behavioral_df, trails_in_block=30
            )
            # print("here")
            processed_behavioral_df = BehavioralUtilities.del_first_row(
                processed_behavioral_df
            )
            
            # print(processed_behavioral_df.to_string())
            BehavioralUtilities.verify_table(processed_behavioral_df)
            new_path = f.replace(".csv", "_processed.csv")
            
            processed_behavioral_df.to_csv(
                new_path,
                index=True,
            )
        except (TypeError, FileNotFoundError) as e:
            print(e)
            pass

        

def one_process():
    
    f = "/media/rory/RDT VIDS/BORIS/RRD168/RDT OPTO CHOICE 0114/RRD168 01142021.csv"
    mouse, date = parse_abet_file(f)

    correction_file = "/media/rory/Padlock_DT/Opto_Analysis/detecting_light_dark_frames/opto_abet_file_corrections.csv"
    correction_file_df = pd.read_csv(correction_file)
    to_add = None
    
    for count, path in enumerate(list(correction_file_df["vid_path"])):
        if mouse in path and date in path:
            to_add = list(correction_file_df["ABET_addition_correction_time_(s)"])[count]
    
    print(f"Adding {to_add} to times")
    
    try:

        print("CURRENT PATH: ", f)

        ABET_1 = BehavioralSession(
            f,
        )
        ABET_1.preprocess_csv()
        df = ABET_1.get_df()

        times = [time + to_add for time in list(df["Evnt_Time"])]
        df["Evnt_Time"] = times

        print(df.head())

        grouped_by_trialnum = df.groupby("trial_num")
        processed_behavioral_df = grouped_by_trialnum.apply(
            BehavioralUtilities.process_csv
        )  # is a new df, it's not the modified df
        processed_behavioral_df = (
            BehavioralUtilities.add_winstay_loseshift_loseomit(
                processed_behavioral_df
            )
        )
        # Add post-hoc processing
        processed_behavioral_df = BehavioralUtilities.shift_col_values(
            processed_behavioral_df
        )

        processed_behavioral_df = BehavioralUtilities.interpolate_block(
            processed_behavioral_df, trails_in_block=30
        )
        # print("here")
        processed_behavioral_df = BehavioralUtilities.del_first_row(
            processed_behavioral_df
        )
        # print(processed_behavioral_df.to_string())
        BehavioralUtilities.verify_table(processed_behavioral_df)
        new_path = f.replace(".csv", "_processed.csv")
        processed_behavioral_df.to_csv(
            new_path,
            index=True,
        )
    except Exception as e:
        print(f"ERROR: {e}")
        pass

if __name__ == "__main__":
    main()
    #one_process()
