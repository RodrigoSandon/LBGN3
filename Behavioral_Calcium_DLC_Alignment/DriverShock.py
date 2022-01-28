from SessionShock import Session
from SessionShock import Neuron
from SessionShock import EventTrace
import time
from pathlib import Path
import os
import glob
from builtins import AttributeError
import sys

import pandas as pd

sys.path.insert(0, "/home/rory/Rodrigo/Behavioral_Calcium_DLC_Analysis")


class Driver:

    def run_one_session_one_neuron():
        list_of_combos_we_care_about = [
            "Bin_Shock Time (s)"
        ]
        try:
            SESSION_PATH = (
                r"/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/Shock Test NEW_SCOPE"
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

                neuron_obj.add_aligned_dff_traces(
                    "Shock Time (s)",
                    half_of_time_window=5,
                    shock_intensity="Bin"
                )
                # time always goes first, everything else goes in order (time window not included in name)
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
                        and "_Shock Time (s)" != eventraces.get_event_traces_name()
                    ):  # omitting an anomaly, we don't want just times to be a grouping
                        is_eventname_in_list_we_care_about = [
                            ele
                            for ele in list_of_combos_we_care_about
                            if (ele == eventraces.get_event_traces_name())
                        ]

                        if bool(is_eventname_in_list_we_care_about) == True:
                            number_of_event_traces += 1

                            eventraces.process_dff_traces_by()  # returns path of csv
                        else:
                            pass
                print("Time taken for %s: %s" %
                      (cell_name, time.time() - start))
                break  # <- FOR RUNNING ONE NEURON
        except Exception as e:
            print(
                "NO ABET TABLE FOUND, SO SINGLE CELL ALIGNMENT & ANALYSIS CAN'T BE DONE!"
            )
            print(e)

    def main():
        """11/12/21 : editing it so it runs through all the sessions in a mouse and ignores the
        sessions in which already have been processed

        Returns:
        dff traces for a given time window for each accepted cell for each session in each mouse, for a given PTP Inscopix folder
        """

        session_types = [
            "Shock Test",
        ]

        list_of_combos_we_care_about = [
            "Bin_Shock Time (s)",
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

                            for cell_name, neuron_obj in session_1.get_neurons().items():
                                print(
                                    "################################ Cell name:",
                                    cell_name,
                                    " ################################",
                                )

                                neuron_obj.add_aligned_dff_traces(
                                    "Shock Time (s)",
                                    half_of_time_window=5,
                                    shock_intensity="Bin"
                                )
                                # time always goes first, everything else goes in order (time window not included in name)
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
                                        "_Choice Time (s)" != eventraces.get_event_traces_name(
                                        )
                                        and "_Start Time (s)" != eventraces.get_event_traces_name()
                                        and "_Collection Time (s)" != eventraces.get_event_traces_name()
                                        and "_Shock Time (s)" != eventraces.get_event_traces_name()
                                    ):  # omitting an anomaly, we don't want just times to be a grouping
                                        is_eventname_in_list_we_care_about = [
                                            ele
                                            for ele in list_of_combos_we_care_about
                                            if (ele == eventraces.get_event_traces_name())
                                        ]

                                        if bool(is_eventname_in_list_we_care_about) == True:
                                            number_of_event_traces += 1

                                            eventraces.process_dff_traces_by()  # returns path of csv
                                        else:
                                            pass
                                print("Time taken for %s: %s" %
                                      (cell_name, time.time() - start))
                        except Exception as e:
                            print(
                                "NO ABET TABLE FOUND, SO SINGLE CELL ALIGNMENT & ANALYSIS CAN'T BE DONE!"
                            )
                            print(e)


if __name__ == "__main__":
    Driver.main()
