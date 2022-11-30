from concurrent.futures import process
from GroupbyAlignmentVelocity_opto import Session
import time
from pathlib import Path
import os
import glob
from builtins import AttributeError
import sys
import subprocess

import pandas as pd

from GroupbyAlignmentVelocity_opto import EventVelocity, Velocity

sys.path.insert(0, "/media/rory/Padlock_DT/Rodrigo/Opto_Speed_Analysis/Analysis")

def find_paths_endswith(root_path, endswith) -> list:

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )

    return files


class Driver:
    def main():

        list_of_combos_we_care_about = [
            
            "Block_Trial_Type_Start_Time_(s)",

        ]

        """list_of_combos_we_care_about = [
            
            "Block_Trial_Type_Start_Time_(s)",
            "Block_Reward_Size_Start_Time_(s)",
            "Block_Reward_Size_Shock_Ocurred_Start_Time_(s)",
            "Block_Trial_Type_Reward_Size_Start_Time_(s)",
            "Block_Trial_Type_Shock_Ocurred_Start_Time_(s)",

        ]"""

        """list_of_combos_we_care_about = [
            "Block_Start_Time_(s)",
            "Block_Omission_Start_Time_(s)",
            "Block_Reward_Size_Start_Time_(s)",
            "Block_Reward_Size_Shock_Ocurred_Start_Time_(s)",
            "Block_Shock_Ocurred_Start_Time_(s)",
            "Block_Trial_Type_Start_Time_(s)",
            "Shock_Ocurred_Start_Time_(s)",
            "Trial_Type_Start_Time_(s)",
            "Trial_Type_Reward_Size_Start_Time_(s)",
            "Block_Trial_Type_Omission_Start_Time_(s)",
            "Block_Trial_Type_Reward_Size_Start_Time_(s)",
            "Block_Trial_Type_Shock_Ocurred_Start_Time_(s)",
            "Block_Trial_Type_Win_or_Loss_Start_Time_(s)",
            "Trial_Type_Shock_Ocurred_Start_Time_(s)",
            "Win_or_Loss_Start_Time_(s)",
            "Block_Win_or_Loss_Start_Time_(s)",
            "Learning_Stratergy_Start_Time_(s)",
            "Omission_Start_Time_(s)",
            "Reward_Size_Start_Time_(s)",
        ]"""

        processed = 0
        session_type = "choice"

        ROOT = r"/media/rory/RDT VIDS/BORIS_merge"

        session_paths = []

        for folder in os.listdir(ROOT):
            if "RRD" in folder and "_" not in folder:
                session_paths.append(os.path.join(ROOT, folder))

        ROOT_2 = r"/media/rory/RDT VIDS/BORIS"

        for folder in os.listdir(ROOT_2):
            if "RRD" in folder and "_" not in folder:
                session_paths.append(os.path.join(ROOT_2, folder))
                                
        session_paths = set(session_paths)
        print("num session paths:",len(session_paths))
        print(session_paths)
        
        start = time.time()
        for count, session_path in enumerate(session_paths):
            print(f"Working on {count}/{len(session_paths)}... {session_path}")

            mouse = session_path.split("/")[-1]
            try:
                try:
                    session_1 = Session(session_path, mouse, session_type)
                except IndexError as e:
                    print(e)

                for (
                    column_focus, vel_obj
                ) in session_1.velocity.items():
                    print(
                        "################################ Column focus:",
                        column_focus,
                        " ################################",
                    )
                    vel_obj: Velocity
                    vel_obj.add_aligned_velocities(
                        "Start_Time_(s)",
                        half_of_time_window=5,
                        block="Block",
                        trial_type="Trial_Type",
                        rew_size="Reward_Size",
                        shock="Shock_Ocurred",
                        omission="Omission",
                        win_loss="Win_or_Loss",
                        learning_strat="Learning_Stratergy",
                    )
                    
                    number_of_event_traces = 0
                    
                    for col_name, event in vel_obj.categorized_vels.items():
                        event: EventVelocity
                    
                        if (
                            "_Start_Time_(s)"
                            != event.event_name
                            and "_Choice_Time_(s)"
                            != event.event_name
                            and "_Collection_Time_(s)"
                            != event.event_name
                        ):  # omitting an anomaly
                            is_eventname_in_list_we_care_about = [
                                ele
                                for ele in list_of_combos_we_care_about
                                if (
                                    ele
                                    == event.event_name
                                )
                            ]

                            if (
                                bool(
                                    is_eventname_in_list_we_care_about)
                                == True
                            ):
                                print(
                                    "Event name: ",
                                    event.event_name,
                                )
                                number_of_event_traces += 1
                                
                                event.process_speed_by()  # returns path of csv
                                
                            else:
                                """print(
                                    f"WE DON'T CARE ABOUT: {event.event_name}"
                                )"""
                                pass
                processed += 1
            except (UnboundLocalError,ValueError, AttributeError, TypeError) as e:
                print(e)
                pass       
        print(
            "Time taken %s"
            % (time.time() - start)
        )
        print(f"num session paths processed: {processed}/{len(session_paths)}")

def count_sessions_processed():
    ROOT = r"/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis_2/"
    session_paths = []

    for file_o_folder in os.listdir(ROOT):
            if os.path.isdir(os.path.join(ROOT, file_o_folder)):
                for folder in os.listdir(os.path.join(ROOT, file_o_folder)):
                    if "CHOICE" in folder:
                        for file in os.listdir(os.path.join(ROOT, file_o_folder, folder)):
                            if ".csv" in file:
                                session_paths.append(os.path.join(ROOT, file_o_folder, folder))
    session_paths = set(session_paths)  
    print("num session paths:",len(session_paths))
    print(*session_paths, sep="\n")

def count_sessions_missing_choice():
    ROOT = r"/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis_2/"
    session_paths = []

    for file_o_folder in os.listdir(ROOT):
        if os.path.isdir(os.path.join(ROOT, file_o_folder)) and "tabulated" not in os.path.join(ROOT, file_o_folder):
            choice_in_folder = False
            for folder in os.listdir(os.path.join(ROOT, file_o_folder)):
                if "CHOICE" in folder:
                    choice_in_folder = True
            if choice_in_folder == False:
                session_paths.append(file_o_folder)

    session_paths = set(session_paths)  
    print("num mice missing their CHOICE session in RDT VIDS hardrive:",len(session_paths))
    print(*session_paths, sep="\n")

def main2():

        list_of_combos_we_care_about = [
            
            "Block_Trial_Type_Start_Time_(s)",

        ]

        """list_of_combos_we_care_about = [
            
            "Block_Trial_Type_Start_Time_(s)",
            "Block_Reward_Size_Start_Time_(s)",
            "Block_Reward_Size_Shock_Ocurred_Start_Time_(s)",
            "Block_Trial_Type_Reward_Size_Start_Time_(s)",
            "Block_Trial_Type_Shock_Ocurred_Start_Time_(s)",

        ]"""

        """list_of_combos_we_care_about = [
            "Block_Start_Time_(s)",
            "Block_Omission_Start_Time_(s)",
            "Block_Reward_Size_Start_Time_(s)",
            "Block_Reward_Size_Shock_Ocurred_Start_Time_(s)",
            "Block_Shock_Ocurred_Start_Time_(s)",
            "Block_Trial_Type_Start_Time_(s)",
            "Shock_Ocurred_Start_Time_(s)",
            "Trial_Type_Start_Time_(s)",
            "Trial_Type_Reward_Size_Start_Time_(s)",
            "Block_Trial_Type_Omission_Start_Time_(s)",
            "Block_Trial_Type_Reward_Size_Start_Time_(s)",
            "Block_Trial_Type_Shock_Ocurred_Start_Time_(s)",
            "Block_Trial_Type_Win_or_Loss_Start_Time_(s)",
            "Trial_Type_Shock_Ocurred_Start_Time_(s)",
            "Win_or_Loss_Start_Time_(s)",
            "Block_Win_or_Loss_Start_Time_(s)",
            "Learning_Stratergy_Start_Time_(s)",
            "Omission_Start_Time_(s)",
            "Reward_Size_Start_Time_(s)",
        ]"""

        processed = 0
        session_type = "choice"
        mouse = "RRD105"
                                
        session_path = f"/media/rory/RDT VIDS/BORIS_merge/{mouse}"
 
        print(f"Working on... {session_path}")

        try:
            try:
                session_1 = Session(session_path, mouse, session_type)
            except IndexError as e:
                print(e)

            for (
                column_focus, vel_obj
            ) in session_1.velocity.items():
                print(
                    "################################ Column focus:",
                    column_focus,
                    " ################################",
                )
                vel_obj: Velocity
                vel_obj.add_aligned_velocities(
                    "Start_Time_(s)",
                    half_of_time_window=5,
                    block="Block",
                    trial_type="Trial_Type",
                    rew_size="Reward_Size",
                    shock="Shock_Ocurred",
                    omission="Omission",
                    win_loss="Win_or_Loss",
                    learning_strat="Learning_Stratergy",
                )
                
                number_of_event_traces = 0
                
                for col_name, event in vel_obj.categorized_vels.items():
                    event: EventVelocity
                
                    if (
                        "_Start_Time_(s)"
                        != event.event_name
                        and "_Choice_Time_(s)"
                        != event.event_name
                        and "_Collection_Time_(s)"
                        != event.event_name
                    ):  # omitting an anomaly
                        is_eventname_in_list_we_care_about = [
                            ele
                            for ele in list_of_combos_we_care_about
                            if (
                                ele
                                == event.event_name
                            )
                        ]

                        if (
                            bool(
                                is_eventname_in_list_we_care_about)
                            == True
                        ):
                            print(
                                "Event name: ",
                                event.event_name,
                            )
                            number_of_event_traces += 1
                            
                            event.process_speed_by()  # returns path of csv
                            
                        else:
                            """print(
                                f"WE DON'T CARE ABOUT: {event.event_name}"
                            )"""
                            pass
            processed += 1
        except (UnboundLocalError,ValueError) as e:
            print(e)
            pass       

if __name__ == "__main__":
    #Driver.main()
    #count_sessions_processed()
    #count_sessions_missing_choice()
    main2()
