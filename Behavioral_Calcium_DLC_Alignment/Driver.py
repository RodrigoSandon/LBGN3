from Session import Session
import time
from pathlib import Path
import os
import glob
from builtins import AttributeError
import sys

import pandas as pd

sys.path.insert(0, "/home/rory/Rodrigo/Behavioral_Calcium_DLC_Analysis")


class Driver:
    def main():
        """11/12/21 : editing it so it runs through all the sessions in a mouse and ignores the
        sessions in which already have been processed

        Returns:
        dff traces for a given time window for each accepted cell for each session in each mouse, for a given PTP Inscopix folder
        """

        session_types = [
            "PR D1",
            "PR D2",
            "Pre-RDT RM",
            "RDT D1",
            "RDT D2",
            "RDT D3",
            "Post-RDT D1",
            "Post-RDT D2",
            "Post-RDT D3",
            "RM D1",
            "RM D2",
            "RM D3",
            "RM D8",
            "RM D9",
            "RM D10",
            "Late Shock D1",
            "Late Shock D2",
        ]  # 1/3/22 ->DONT INCLUDE SHOCK SESSIONS IN THIS PROCESS

        list_of_combos_we_care_about = [
            "Block_Choice Time (s)",
            "Block_Learning Stratergy_Choice Time (s)",
            "Block_Omission_Choice Time (s)",
            "Block_Reward Size_Choice Time (s)",
            "Block_Reward Size_Shock Ocurred_Choice Time (s)",
            "Block_Shock Ocurred_Choice Time (s)",
            "Block_Trial Type_Choice Time (s)",
            "Shock Ocurred_Choice Time (s)",
            "Trial Type_Choice Time (s)",
            "Trial Type_Reward Size_Choice Time (s)",
            "Block_Trial Type_Omission_Choice Time (s)",
            "Block_Trial Type_Reward Size_Choice Time (s)",
            "Block_Trial Type_Shock Ocurred_Choice Time (s)",
            "Block_Trial Type_Win or Loss_Choice Time (s)",
            "Trial Type_Shock Ocurred_Choice Time (s)",
            "Win or Loss_Choice Time (s)",
            "Block_Win or Loss_Choice Time (s)",
            "Learning Stratergy_Choice Time (s)",
            "Omission_Choice Time (s)",
            "Reward Size_Choice Time (s)",
        ]

        MASTER_ROOT = r"/media/rory/Padlock_DT/BLA_Analysis/"
        for folder_name in os.listdir(MASTER_ROOT):
            print(folder_name)
            if "PTP" in folder_name:
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

                            # now make individual neuron objects for each of the columns
                            # print("Dict of all neurons in session: ", session_1.neurons)

                            # Go into one neuron obj from the neuron dict, call its method that returns a list
                            # but only getting values 0-10 (exclusive)

                            # Looping into all neuron objects in neurons dict from session
                            for (
                                cell_name,
                                neuron_obj,
                            ) in session_1.get_neurons().items():
                                print(
                                    "################################ Cell name:",
                                    cell_name,
                                    " ################################",
                                )

                                # print(neuron_obj.get_sample_dff_times())
                                # print(neuron_obj.get_dff_trace())
                                """            neuron_obj.add_aligned_dff_traces(
                                        "Choice Time (s)",
                                        half_of_time_window=10,
                                        trial_type="Trial Type",
                                        reward_size="Reward Size",
                                    )"""
                                # 12/22/21 <- how to align shock sessions?
                                neuron_obj.add_aligned_dff_traces(
                                    "Choice Time (s)",
                                    half_of_time_window=10,
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
                                    print(
                                        "Event traces name: ",
                                        eventraces.get_event_traces_name(),
                                    )
                                    if (
                                        "_Choice Time (s)"
                                        != eventraces.get_event_traces_name()
                                        and "_Start Time (s)"
                                        != eventraces.get_event_traces_name()
                                        and "_Collection Time (s)"
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
                                            """print(
                                                    "Event trace number: ",
                                                    number_of_event_traces,
                                                )"""
                                            # print(eventraces.get_dff_traces_of_neuron())
                                            # but can it pull the abet data for every event trace?
                                            # print(eventraces.get_abet())
                                            """now I have abet and dff ready to go, now write
                                                a function in EventTraces to make this processed table
                                                for this neuron depending on the input parameters"""
                                            # testing groupby

                                            eventraces.process_dff_traces_by()  # returns path of csv
                                            # avg_cell_eventrace(csv_path)
                                            # PLOT
                                        else:
                                            """print(
                                                f"WE DON'T CARE ABOUT: {eventraces.get_event_traces_name()}"
                                            )"""
                                            pass
                                print(
                                    "Time taken for %s: %s"
                                    % (cell_name, time.time() - start)
                                )
                        except:
                            print(
                                "NO ABET TABLE FOUND, SO SINGLE CELL ALIGNMENT & ANALYSIS CAN'T BE DONE!"
                            )
                            pass

        # Does it identify the ABET file for this session? yes
        # print(session_1.behavioral_df.head())

    def run_one_session_one_neuron():
        list_of_combos_we_care_about = [
            "Block_Choice Time (s)",
            "Block_Learning Stratergy_Choice Time (s)",
            "Block_Omission_Choice Time (s)",
            "Block_Reward Size_Choice Time (s)",
            "Block_Reward Size_Shock Ocurred_Choice Time (s)",
            "Block_Shock Ocurred_Choice Time (s)",
            "Block_Trial Type_Choice Time (s)",
            "Shock Ocurred_Choice Time (s)",
            "Trial Type_Choice Time (s)",
            "Trial Type_Reward Size_Choice Time (s)",
            "Block_Trial Type_Omission_Choice Time (s)",
            "Block_Trial Type_Reward Size_Choice Time (s)",
            "Block_Trial Type_Shock Ocurred_Choice Time (s)",
            "Block_Trial Type_Win or Loss_Choice Time (s)",
            "Trial Type_Shock Ocurred_Choice Time (s)",
            "Win or Loss_Choice Time (s)",
            "Block_Win or Loss_Choice Time (s)",
            "Learning Stratergy_Choice Time (s)",
            "Omission_Choice Time (s)",
            "Reward Size_Choice Time (s)",
        ]
        try:
            SESSION_PATH = (
                r"/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/RM D2"
            )

            session_1 = Session(SESSION_PATH)

            # now make individual neuron objects for each of the columns
            # print("Dict of all neurons in session: ", session_1.neurons)

            # Go into one neuron obj from the neuron dict, call its method that returns a list
            # but only getting values 0-10 (exclusive)

            # Looping into all neuron objects in neurons dict from session
            for cell_name, neuron_obj in session_1.get_neurons().items():
                print(
                    "################################ Cell name:",
                    cell_name,
                    " ################################",
                )

                # print(neuron_obj.get_sample_dff_times())
                # print(neuron_obj.get_dff_trace())
                """            neuron_obj.add_aligned_dff_traces(
                        "Choice Time (s)",
                        half_of_time_window=10,
                        trial_type="Trial Type",
                        reward_size="Reward Size",
                    )"""
                neuron_obj.add_aligned_dff_traces(
                    "Choice Time (s)",
                    half_of_time_window=10,
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
                    print(
                        "Event traces name: ",
                        eventraces.get_event_traces_name(),
                    )
                    if (
                        "_Choice Time (s)" != eventraces.get_event_traces_name()
                        and "_Start Time (s)" != eventraces.get_event_traces_name()
                        and "_Collection Time (s)" != eventraces.get_event_traces_name()
                    ):  # omitting an anomaly
                        is_eventname_in_list_we_care_about = [
                            ele
                            for ele in list_of_combos_we_care_about
                            if (ele == eventraces.get_event_traces_name())
                        ]

                        if bool(is_eventname_in_list_we_care_about) == True:
                            """print(
                                f"WE CARE ABOUT: {eventraces.get_event_traces_name()}"
                            )"""
                            number_of_event_traces += 1
                            """print(
                                    "Event trace number: ",
                                    number_of_event_traces,
                                )"""
                            # print(eventraces.get_dff_traces_of_neuron())
                            # but can it pull the abet data for every event trace?
                            # print(eventraces.get_abet())
                            """now I have abet and dff ready to go, now write
                                a function in EventTraces to make this processed table
                                for this neuron depending on the input parameters"""
                            # testing groupby

                            eventraces.process_dff_traces_by()  # returns path of csv
                            # avg_cell_eventrace(csv_path)
                            # PLOT
                        else:
                            """print(
                                f"WE DON'T CARE ABOUT: {eventraces.get_event_traces_name()}"
                            )"""
                            pass
                print("Time taken for %s: %s" %
                      (cell_name, time.time() - start))
                break  # <- FOR RUNNING ONE NEURON
        except Exception as e:
            print(
                "NO ABET TABLE FOUND, SO SINGLE CELL ALIGNMENT & ANALYSIS CAN'T BE DONE!"
            )
            print(e)


if __name__ == "__main__":
    Driver.main()
    # Driver.run_one_session_one_neuron()
