from Session_custom_timewindow_aligntochoice import Session
import time
from pathlib import Path
import os
import glob
from builtins import AttributeError
import sys

import pandas as pd

sys.path.insert(0, "/home/rory/Rodrigo/Behavioral_Calcium_DLC_Analysis")

def main():

    session_types = [

        "Pre-RDT RM",
        "RDT D1",
        "RDT D2",
        "RDT D3"

    ]  # 1/3/22 ->DONT INCLUDE SHOCK SESSIONS IN THIS PROCESS

    list_of_combos_we_care_about = [
        "Block_Start Time (s)_Collection Time (s)",
        "Block_Learning Stratergy_Start Time (s)_Collection Time (s)",
        "Block_Reward Size_Start Time (s)_Collection Time (s)",
        "Block_Reward Size_Shock Ocurred_Start Time (s)_Collection Time (s)",
        "Block_Shock Ocurred_Start Time (s)_Collection Time (s)",
        "Block_Trial Type_Start Time (s)_Collection Time (s)",
        "Shock Ocurred_Start Time (s)_Collection Time (s)",
        "Trial Type_Start Time (s)_Collection Time (s)",
        "Trial Type_Reward Size_Start Time (s)_Collection Time (s)",
        "Block_Trial Type_Reward Size_Start Time (s)_Collection Time (s)",
        "Block_Trial Type_Shock Ocurred_Start Time (s)_Collection Time (s)",
        "Block_Trial Type_Win or Loss_Start Time (s)_Collection Time (s)",
        "Trial Type_Shock Ocurred_Start Time (s)_Collection Time (s)",
        "Win or Loss_Start Time (s)_Collection Time (s)",
        "Block_Win or Loss_Start Time (s)_Collection Time (s)",
        "Learning Stratergy_Start Time (s)_Collection Time (s)",
        "Reward Size_Start Time (s)_Collection Time (s)",
    ] #omitting omission events for now

    MASTER_ROOT = r"/media/rory/Padlock_DT/BLA_Analysis/"
    folder_names = ["PTP_Inscopix_#1", "PTP_Inscopix_#3", "PTP_Inscopix_#4", "PTP_Inscopix_#5"]
    for folder_name in folder_names:
        BATCH_ROOT = os.path.join(MASTER_ROOT, folder_name)
        mouse_paths = [
            os.path.join(BATCH_ROOT, dir)
            for dir in os.listdir(BATCH_ROOT)
            if os.path.isdir(os.path.join(BATCH_ROOT, dir))
            and dir.startswith("BLA")
        ]
        for mouse_path in mouse_paths:
            print("CURRENT MOUSE PATH: ", mouse_path)

            session_paths = []

            MOUSE_PATH = Path(mouse_path)
            for root, dirs, files in os.walk(MOUSE_PATH):
                for dir_name in dirs:
                    for ses_type in session_types:
                        if (
                            dir_name.find(ses_type) != -1
                        ):  # means ses type string was found in dirname
                            SESSION_PATH = os.path.join(root, dir_name)
                            session_paths.append(SESSION_PATH)

            # TODO: FIND SESSION PATHS
            print("FOUND SESSION PATHS FOR MOUSE:")
            print(session_paths)
            for session_path in session_paths:
                print(f"Working on... {session_path}")
                try:

                    session_1 = Session(session_path)

                    count = 0
                    for (cell_name,neuron_obj) in session_1.get_neurons().items():
                            count +=1
                            print(
                                "################################ Cell name:",
                                cell_name,
                                " ################################",
                            )

                            
                            neuron_obj.add_aligned_dff_traces(
                                "Start Time (s)",
                                "Collection Time (s)",
                                block="Block",
                                trial_type="Trial Type",
                                rew_size="Reward Size",
                                shock="Shock Ocurred",
                                omission="Omission",
                                win_loss="Win or Loss",
                                learning_strat="Learning Stratergy",
                            )
                            # time always goes last, everything else goes in order (time window not included in name)
                            # print(neuron_obj.categorized_dff_traces)
                            number_of_event_traces = 0
                            start = time.time()
                            for (
                                event_name,
                                eventraces,
                            ) in neuron_obj.get_categorized_dff_traces().items():
                                """print(
                                    "Event traces name: ",
                                    eventraces.get_event_traces_name(),
                                )"""
                                if (
                                    "_Start Time (s)_Collection Time (s)"
                                    != eventraces.get_event_traces_name()
                                ):  # omitting an anomaly
                                    is_eventname_in_list_we_care_about = [
                                        ele
                                        for ele in list_of_combos_we_care_about
                                        if (
                                            ele
                                            == eventraces.get_event_traces_name()
                                        )
                                    ]

                                    if (
                                        bool(
                                            is_eventname_in_list_we_care_about)
                                        == True
                                    ):
                                        """print(
                                            f"WE CARE ABOUT: {eventraces.get_event_traces_name()}"
                                        )"""
                                        number_of_event_traces += 1
                                        
                                        eventraces.process_dff_traces_by()  # returns path of csv
                                        # avg_cell_eventrace(csv_path)
                                        # PLOT
                                    else:
                        
                                        pass
                            print(
                                "Time taken for %s: %s"
                                % (cell_name, time.time() - start)
                            )
                except Exception as e:
                    print("EXCEPTION:", e)
                    print(
                        "NO ABET TABLE FOUND, SO SINGLE CELL ALIGNMENT & ANALYSIS CAN'T BE DONE!"
                    )
                    pass

def run_one_session():
    list_of_combos_we_care_about = [
        "Shock Ocurred_Start Time (s)_Collection Time (s)",
    ]
    
    SESSION_PATH = (
        r"/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-9/RDT D1"
    )

    session_1 = Session(SESSION_PATH)

    count = 0
    for (cell_name,neuron_obj) in session_1.get_neurons().items():
        count +=1
        print(
            "################################ Cell name:",
            cell_name,
            " ################################",
        )

        
        neuron_obj.add_aligned_dff_traces(
            "Start Time (s)",
            "Collection Time (s)",
            block="Block",
            trial_type="Trial Type",
            rew_size="Reward Size",
            shock="Shock Ocurred",
            omission="Omission",
            win_loss="Win or Loss",
            learning_strat="Learning Stratergy",
        )
        # time always goes last, everything else goes in order (time window not included in name)
        # print(neuron_obj.categorized_dff_traces)
        number_of_event_traces = 0
        start = time.time()
        for (
            event_name,
            eventraces,
        ) in neuron_obj.get_categorized_dff_traces().items():
            """print(
                "Event traces name: ",
                eventraces.get_event_traces_name(),
            )"""
            if (
                "_Start Time (s)_Collection Time (s)"
                != eventraces.get_event_traces_name()
            ):  # omitting an anomaly
                is_eventname_in_list_we_care_about = [
                    ele
                    for ele in list_of_combos_we_care_about
                    if (
                        ele
                        == eventraces.get_event_traces_name()
                    )
                ]

                if (
                    bool(
                        is_eventname_in_list_we_care_about)
                    == True
                ):
                    """print(
                        f"WE CARE ABOUT: {eventraces.get_event_traces_name()}"
                    )"""
                    number_of_event_traces += 1
                    
                    eventraces.process_dff_traces_by()  # returns path of csv
                    # avg_cell_eventrace(csv_path)
                    # PLOT
                else:
    
                    pass
        print(
            "Time taken for %s: %s"
            % (cell_name, time.time() - start)
        )
        if count == 2:
            break  # <- FOR RUNNING ONE NEURON


if __name__ == "__main__":
    main()
    #run_one_session()