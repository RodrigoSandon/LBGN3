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


def main():

    ROOT = r"/media/rory/RDT VIDS/ABET_files_opto/"

    DST_1 = r"/media/rory/RDT VIDS/BORIS_merge"
    DST_2 = r"/media/rory/RDT VIDS/BORIS"
    files = find_paths_startswith_and_endswith(ROOT, "RRD", ".csv")

    for count, f in enumerate(files):
        print(f"file {count}/{len(files)}")
        print("CURRENT PATH: ", f)

        try:
            file_name = f.split("/")[-1]

            mouse, date = parse_abet_file(f)

            # check if mouse exists in either boris
            if os.path.isdir(f"{DST_1}/{mouse}"):
                f = f"{DST_1}/{mouse}/{file_name}"

            elif os.path.isdir(f"{DST_2}/{mouse}"):
                f = f"{DST_2}/{mouse}/{file_name}"
            #dst_abet_dir = f"/media/rory/RDT VIDS/BORIS_merge/{mouse}"

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
        except (TypeError, FileNotFoundError, NotADirectoryError) as e:
            print(e)
            pass

        

def one_process():
    
    f = "/media/rory/RDT VIDS/BORIS_merge/RRD76/RRD76 10222019.csv"
    mouse, date = parse_abet_file(f)
    print(mouse)
    print(date)

    correction_file = "/media/rory/Padlock_DT/Opto_Speed_Analysis/detecting_light_dark_frames/opto_abet_file_corrections.csv"
    correction_file_df = pd.read_csv(correction_file)
    to_add = None
    
    for count, path in enumerate(list(correction_file_df["vid_path"])):
        if mouse in path and date in path:
            to_add = list(correction_file_df["ABET_addition_correction_time_(s)"])[count]
    
    print(f"Adding {to_add} to times")


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

if __name__ == "__main__":
    #main()
    one_process()
