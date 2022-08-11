from multiprocessing import Event
import pandas as pd
import numpy as np
from pathlib import Path
import os
import glob
import Utilities_opto
from typing import List
from itertools import combinations


class Session(object):

    velocity = {}

    def __init__(self, session_path):
        
        Session.session_path = session_path
        Session.behavioral_df = self.load_table("abet")
        Session.sleap_df = self.load_table("sleap")

        Session.velocity = self.parse_sleap_table()

    def load_table(self, table_to_extract):
        endswith = None
        message = None

        if table_to_extract == "dff":
            endswith = "processed_dff_and_body_data.csv"
            message = "No dff table found!"
        elif table_to_extract == "abet":
            endswith = "_processed.csv"
            message = "No ABET table found!"
        elif table_to_extract == "sleap":
            endswith = "body_sleap_data.csv"
            message = "No SLEAP table found!"

        path = Utilities_opto.find_paths_endswith(
            self.session_path, endswith
        )
        if path == None:
            print(message)
            return None
        table = pd.read_csv(path[0])

        return table

    def parse_sleap_table(self):
        velocity_info = {}

        time_vel = Velocity(list(self.sleap_df["idx_time"]),
                            list(self.sleap_df["vel_cm_s"]))
        velocity_info["time_vel"] = time_vel

        return velocity_info

class Velocity(Session):
    """Want to bin velocities, each velocity has many subcombos it belongs to"""

    categorized_vels = {}  # [Event name + number] : EventTraces

    def __init__(self, time: list, speed: list):
        self.time = time
        self.speed = speed

    # creates an even trace for all given combinations of the list of values inputted
    def add_aligned_velocities(
        self,
        acquire_time,
        half_of_time_window,
        **groupby_dict
    ):
        """**kwargs takes in named variables we want to groupby"""
        event_name_list = []
        for groupby_key, value in groupby_dict.items():
            event_name_list.append(value)

        number_items_to_select = list(range(len(event_name_list) + 1))
        for i in number_items_to_select:
            # for each r allowed, make all possible combos (r=1,2,3,...,7)
            to_select = i
            combs = combinations(event_name_list, to_select)
            for combine_by_list in list(combs):
                #for each combo, there's a name and new obj you make w all the info
                event_name = (
                    "_".join(combine_by_list)
                    + "_"
                    + acquire_time
                )

                self.categorized_vels[event_name] = EventVelocity(
                    self.speed,
                    event_name,
                    acquire_time,
                    half_of_time_window,
                    list(combine_by_list),
                )

class EventVelocity(Velocity):  # for one combo
    # don't need a time list?
    # these are for getting a table of all events that happened under this one subevent
    # FOR SPEED
    all_speedlists_foracombo_eventcomboname_dict = {} # structure is in 2d

    # FOR SPEED
    avg_speedlists_foracombo_eventcomboname_dict = {} # structure is in 1d

    def __init__(
        self,
        speed,
        event_name,
        acquire_time,
        half_of_time_window,
        groupby_list: list,
    ):
        
        self.speed = speed
        self.event_name = event_name
        self.acquire_time = acquire_time
        self.half_of_time_window = half_of_time_window
        self.groupby_list = groupby_list
        self.events_omitted = 0
        #super().__init__(speed)

    def find_idx_of_time_bound(self, time):
        
        idx = Utilities_opto.binary_search(
            list(self.sleap_df["idx_time"]), time
        )
        idx_time_val = self.sleap_df.iloc[
            idx, self.sleap_df.columns.get_loc("idx_time")
        ]

        return idx, idx_time_val

    def create_all_events_2d_lists(self, list_of_idxs, start_choice_collect):
        """This is for one grouping found from the groupby columns"""
        
        list_of_events_speed = []

        for abet_idx in list_of_idxs:
            # abet_idx = abet_idx - 1 Omitting this line made it so identified all events for each cell properly, I wonder why? 11/5/21
            # - 1 BECAUSE WE WANT IT TO START AT 0, BECAUSE INDICES SHIFTED UP 1 WHEN DELETING FIRST EMPTY COLUMN
            time_for_this_idx_in_abet = self.behavioral_df.iloc[
                abet_idx, self.behavioral_df.columns.get_loc(start_choice_collect)
            ]
            if (
                str(time_for_this_idx_in_abet) != "nan"
            ):  # if the time is nan, then we don't include it in the stack of dff traces

                lower_bound_time = time_for_this_idx_in_abet - self.half_of_time_window
                upper_bound_time = time_for_this_idx_in_abet + self.half_of_time_window

                idx_df_lower_bound_time, lower_time_val = self.find_idx_of_time_bound(
                    lower_bound_time
                )
                """print(
                    (
                        "desired START time to extract dff traces from: %s, actual time extracted: %s"
                    )
                    % (lower_bound_time, lower_time_val)
                )"""
                idx_df_upper_bound_time, upper_time_val = self.find_idx_of_time_bound(
                    upper_bound_time
                )
                """print(
                    (
                        "desired END time to extract dff traces from: %s, actual time extracted: %s"
                    )
                    % (upper_bound_time, upper_time_val)
                )"""
                
                speed_block = self.speed[
                        idx_df_lower_bound_time:idx_df_upper_bound_time
                    ]

                list_of_events_speed.append(speed_block)
                
            else:
                self.events_omitted += 1
                pass

        return list_of_events_speed  # this is a 2d list - SCOPE OF THIS WAS INNER

    def trim_grouped_df(self, grouped_df):
        """Drops any columns that are past the half_the_time_window *10*2 - 1"""
        trunc_df = grouped_df.truncate(
            after=(self.half_of_time_window *30 * 2 - 1), axis=1
        )
        return trunc_df

    def get_xaxis_list_for_plotting(self):
        """Hertz of frames is 10 Hz, so increment by 0.1 within this time window.

        Returns: path to where csv was saved."""
        return np.arange(
            -1 * (self.half_of_time_window), self.half_of_time_window, 0.03333
        ).tolist()

    def process_speed_by(self):

        grouped_table = self.behavioral_df.groupby(self.groupby_list)

        x_axis = self.get_xaxis_list_for_plotting()

        # SUBCOMBO PROCESSING
        for key, val in grouped_table.groups.items():

            if "nan" not in str(key):

                number_of_event_appearances = len(list(val))
                print(key, ": ", list(val))
                
                self.all_speedlists_foracombo_eventcomboname_dict[
                    key
                    ] = self.create_all_events_2d_lists(
                    list(val), self.acquire_time
                )

                group_df_speed = pd.DataFrame.from_records(
                    self.all_speedlists_foracombo_eventcomboname_dict[key]
                )

                group_df_speed = self.trim_grouped_df(group_df_speed)

                # Doing some editing on this df
                group_df_speed = Utilities_opto.rename_all_col_names(group_df_speed, x_axis)

                group_df_speed.insert(
                    loc=0,
                    column="Event_#",
                    value=Utilities_opto.make_value_list_for_col(
                        "Event", number_of_event_appearances - self.events_omitted
                    ),
                )

                combo_name = str(key)

                new_path = os.path.join(
                    self.session_path,
                    "AlignmentData",
                    self.event_name,
                    combo_name,
                )
                # Insert to aligned dff dict that corresponds to this object
                # self.aligned_dff_dict[self.get_event_traces_name] = group_df_dff

                os.makedirs(new_path, exist_ok=True)

                name_of_speed_df = "speeds.csv"
                csv_path_speed = os.path.join(new_path, name_of_speed_df)
                group_df_speed.to_csv(csv_path_speed, index=False)
                print(group_df_speed.head())

                event_num = len(group_df_speed)

                ### Add on analysis here ###
                
                Utilities_opto.plot_indv_speeds(csv_path_speed, name_of_speed_df)
                avg_speed_csv_path = Utilities_opto.make_avg_speed_table(name_of_speed_df, csv_path_speed, self.half_of_time_window)
                Utilities_opto.plot_avg_speed(avg_speed_csv_path, event_num)

                # make sure the events omitted resets after ever subcombo within an eventtrace
                self.events_omitted = 0
            else:
                print(f"WILL NOT INCLUDE {str(key)}")
