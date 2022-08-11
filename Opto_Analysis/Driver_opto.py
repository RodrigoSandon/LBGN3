from GroupbyAlignmentVelocity_opto import Session
import time
from pathlib import Path
import os
import glob
from builtins import AttributeError
import sys

import pandas as pd

from GroupbyAlignmentVelocity_opto import EventVelocity, Velocity

sys.path.insert(0, "/media/rory/Padlock_DT/Rodrigo/Opto_Analysis")

def find_sleap_data():
    pass

def find_abet():
    pass

class Driver:
    def main():

        list_of_combos_we_care_about = [
            "Block_Learning_Stratergy_Start_Time_(s)",
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
            "Block_Win_or_Loss_Start_Time_(s)",
            "Omission_Start_Time_(s)",
            "Reward_Size_Start_Time_(s)",
        ]

        #Right now one session at a time
        session_paths = ["/media/rory/RDT VIDS/BORIS/RRD171/RDT OPTO CHOICE 0104"]
        
        start = time.time()
        for session_path in session_paths:
            print(f"Working on... {session_path}")

            session_1 = Session(session_path)

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
        print(
            "Time taken %s"
            % (time.time() - start)
        )

if __name__ == "__main__":
    Driver.main()
    #Driver.run_one_session_one_neuron()
