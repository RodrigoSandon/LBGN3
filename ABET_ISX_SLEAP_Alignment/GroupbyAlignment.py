import pandas as pd
import numpy as np
from pathlib import Path
import os
import glob
import Utilities
from typing import List
from itertools import combinations


class Session(object):
    """Loading requires that you have dff, abet, and dlc data all in one session folder already.

    Returns:
        dff traces for a given time window for each accepted cell in the session.
    """

    neurons = {}  # [cell_name] = Neuron
    # dff_times = []  # list of times, length same as length of dff traces for all cells
    """Do not need this^ variable anymore"""

    def __init__(self, session_path):
        # /media/rory/PTP Inscopix 2/PTP_Inscopix_#3/BLA-Insc-6/Session-20210518-102215_BLA-Insc-6_RDT_D1
        # new: /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/RDT D2/2021-06-03-11-20-22_video_BLA-Insc-6_RDT_D2_NEW_SCOPE
        Session.session_path = session_path
        # loading options: "dff" "dlc" "abet"
        Session.dff_traces = self.load_table("dff")
        Session.behavioral_df = self.load_table("abet")

        Session.neurons = self.parse_dff_table()

    def parse_dff_table(self):  # give a mini dff trace df for each neuron
        neurons = {}
        # First create Neuron objects for each cell column
        d = self.get_dff_traces().to_dict("list")
        # don't need to remove time anymore
        # print("dict keys: ", d.keys())
        """
        Example:
        d = {
            "Time(s) : [time]
            "COO" : [dff traces]
            "CO1" : [dff traces]
        }
        """

        for cell_name, dff_traces in d.items():
            if cell_name != "Time(s)" and cell_name != "Speed(cm/s)":
                # INFORMATION FOR JUST ONE NEURON: ABS TIME, DFF, N VEL
                sub_dict_for_neuron = {
                    "Time(s)": d["Time(s)"], 
                    cell_name: dff_traces}

                speed = d["Speed(cm/s)"]
                sub_df_for_neuron = pd.DataFrame.from_dict(sub_dict_for_neuron)
                neuron_obj = Neuron(cell_name, sub_df_for_neuron, speed)
                neurons[cell_name] = neuron_obj

        return neurons

    """ Loading DFF, DLC, and ABET tables onto session"""

    def load_table(self, table_to_extract):
        if table_to_extract == "dff":
            path = Utilities.find_dff_trace_path(
                self.session_path, "processed_dff_and_body_data.csv"
            )
            if path == None:
                print("No dff table found!")
                return None
            dff_traces = pd.read_csv(path)
            return dff_traces
            
        elif table_to_extract == "abet":
            path = Utilities.find_dff_trace_path(
                self.session_path, "_ABET_GPIO_processed.csv"
            )
            if path == None:
                print("No ABET table found!")
                return None
            behavioral_df = pd.read_csv(path)

            return behavioral_df

    def get_neurons(self) -> dict:
        return self.neurons

    def get_dff_traces(self) -> pd.DataFrame:
        return self.dff_traces

    def get_dlc_df(self):
        return self.dlc_df

    def get_behavioral_df(self):
        return self.behavioral_df[1:]


class Neuron(Session):
    """A neuron has corresponding dff traces for a given session.
    All neurons within session are under the same time scale"""

    categorized_dff_traces = {}  # [Event name + number] : EventTraces

    def __init__(self, cell_name: str, dff_trace_df: pd.DataFrame, speed: list):
        self.cell_name = cell_name
        self.dff_trace = dff_trace_df
        self.speed = speed

    def get_cell_name(self):
        return self.cell_name

    def set_cell_name(self, new_name):
        self.cell_name = new_name

    def get_dff_trace(self):
        return self.dff_trace

    def get_categorized_dff_traces(self):
        return self.categorized_dff_traces

    # creates an even trace for all given combinations of the list of values inputted
    def add_aligned_dff_traces(
        self,
        acquire_by_start_choice_or_collect_times,
        half_of_time_window,
        **groupby_dict
    ):
        """**kwargs takes in named variables we want to groupby"""
        event_name_list = []
        for groupby_key, value in groupby_dict.items():
            event_name_list.append(value)

        number_items_to_select = list(range(len(event_name_list) + 1))
        for i in number_items_to_select:
            to_select = i
            combs = combinations(event_name_list, to_select)
            for combine_by_list in list(combs):
                # print("curr combo: ", combine_by_list)
                event_name = (
                    "_".join(combine_by_list)
                    + "_"
                    + acquire_by_start_choice_or_collect_times
                )
                self.categorized_dff_traces[event_name] = list(
                    combine_by_list)  # can be omitted

                self.categorized_dff_traces[event_name] = EventTraces(
                    self.cell_name,
                    self.dff_trace,
                    self.speed,
                    event_name,
                    acquire_by_start_choice_or_collect_times,
                    half_of_time_window,
                    list(combine_by_list),
                )


class EventTraces(Neuron):  # for one combo
    """
    Even table defines which combination of column values we are extracting dff traces from.
    For now, there is a focus on alignment based on one combination chosen (multiple columns chosen
    or just 1 column chosen to groupby)
    """
    #these are for getting a table of all events that happened under this one subevent
    # FOR DFF
    all_eventraces_foracombo_eventcomboname_dict = {}  # structure is in 2d
    # FOR SPEED
    all_speedlists_foracombo_eventcomboname_dict = {} # structure is in 2d
    """
    For 1 cell ( a dict of 2d lists)

    event_combo_name : [[event occurance 1 dff traces],
                  [event occurance 2 dff traces]],
    event_combo_name : [[event occurance 1 dff traces],
                  [event occurance 2 dff traces]]
    """
    # FOR DFF
    avg_eventraces_foracombo_eventcomboname_dict = {}  # structure is in 1d

    # FOR SPEED
    avg_speedlists_foracombo_eventcomboname_dict = {} # structure is in 1d
    """
    For 1 cell

    event_combo_name : [avg event dff trace],
    event_combo_name : [avg event dff trace]
    """

    def __init__(
        self,
        cell_name,
        dff_trace,
        speed,
        eventtraces_name,
        start_choice_or_collect_times,
        half_of_time_window,
        groupby_list: list,
    ):
        # so this EventTraces obj should have a name
        self.speed = speed
        self.name = eventtraces_name
        self.start_choice_or_collect_times = start_choice_or_collect_times
        self.half_of_time_window = half_of_time_window
        self.groupby_list = groupby_list
        self.events_omitted = 0
        super().__init__(cell_name, dff_trace, speed)

    def get_event_traces_name(self):
        return self.name

    def get_dff_traces_of_neuron(self):
        return self.get_dff_trace()

    def get_abet(self):
        return self.behavioral_df  # the first row has been deleted

    """TODO: Add another parameter to this function, the dlc data, eventually"""

    def find_idx_of_time_bound(self, time):
        # get col of time as a list for the df of neuron
        idx = Utilities.binary_search(
            list(self.get_dff_traces_of_neuron()["Time(s)"]), time
        )
        idx_time_val = self.get_dff_traces_of_neuron().iloc[
            idx, self.get_dff_traces_of_neuron().columns.get_loc("Time(s)")
        ]

        return idx, idx_time_val

    def create_all_events_2d_lists(self, list_of_idxs, start_choice_collect):
        """This is for one grouping found from the groupby columns"""
        # print("Chosen time to extract: ", start_choice_collect)
        # print("list of idxs for group ", list_of_idxs)  # - works
        list_of_events_dff = []  # inserting only subwindows of dff traces here around a specific time
        # list of dff traces (which is a list), every list within list represents the event found
        list_of_events_speed = []

        for abet_idx in list_of_idxs:
            # abet_idx = abet_idx - 1 Omitting this line made it so identified all events for each cell properly, I wonder why? 11/5/21
            # - 1 BECAUSE WE WANT IT TO START AT 0, BECAUSE INDICES SHIFTED UP 1 WHEN DELETING FIRST EMPTY COLUMN
            time_for_this_idx_in_abet = self.get_abet().iloc[
                abet_idx, self.get_abet().columns.get_loc(start_choice_collect)
            ]
            if (
                str(time_for_this_idx_in_abet) != "nan"
            ):  # if the time is nan, then we don't include it in the stack of dff traces

                # now have time, pull this time from dff of neuron
                """TODO: there could be where the desired time window doesn't include that index"""
                # essentially need to get the indices of what this time range entails
                lower_bound_time = time_for_this_idx_in_abet - self.half_of_time_window
                upper_bound_time = time_for_this_idx_in_abet + self.half_of_time_window

                # I know an exact minus half time window won't exist for this choice time
                # Therefore, we need to do a closest match binary search
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
                # Where taking the subwindow happens
                dff_block_of_neuron = list(
                    self.get_dff_traces_of_neuron()[
                        idx_df_lower_bound_time:idx_df_upper_bound_time
                    ][self.cell_name]
                )

                speed_block_of_neuron = self.speed[
                        idx_df_lower_bound_time:idx_df_upper_bound_time
                    ]

                # print(dff_block_of_neuron) - only one dff block, so something wrong upstream
                list_of_events_dff.append(dff_block_of_neuron)
                list_of_events_speed.append(speed_block_of_neuron)
                # only getting the dff, not considering the relative time, just absolute time
                # now append this
            else:
                self.events_omitted += 1
                pass
        return list_of_events_dff, list_of_events_speed  # this is a 2d list - SCOPE OF THIS WAS INNER

    def trim_grouped_df(self, grouped_df):
        """Drops any columns that are past the half_the_time_window *10*2 - 1"""
        trunc_df = grouped_df.truncate(
            after=(self.half_of_time_window * 10 * 2 - 1), axis=1
        )
        return trunc_df

    def get_xaxis_list_for_plotting(self):
        """Hertz of frames is 10 Hz, so increment by 0.1 within this time window.

        Returns: path to where csv was saved."""
        return np.arange(
            -1 * (self.half_of_time_window), self.half_of_time_window, 0.1
        ).tolist()

    def process_dff_traces_by(self):
        # print("Currently processing dff traces for all groups...")

        # print("groupby list: ", self.groupby_list)

        grouped_table = self.get_abet().groupby(self.groupby_list)
        # print("abet file: ", self.get_abet().head())
        # print(grouped_table.groups)
        # print(type(grouped_table.groups))

        # Now have list o what to group by, say: forced small
        # Now need to find a way to call dff traces of neuron, but needs to be in df data type
        # for every group, found by groupby, index into key and or every element (index of abet table)
        # and make a new table out of it
        # make prettydict to dict first?

        x_axis = self.get_xaxis_list_for_plotting()
        # print(x_axis)
        # print(len(x_axis))

        # SUBCOMBO PROCESSING
        for key, val in grouped_table.groups.items():
            # make sure to not include subcombos that have nans in it
            # val are a list of idx for when to get these events
            # of when choice time occurred

            if "nan" not in str(key):

                number_of_event_appearances = len(list(val))
                print(key, ": ", list(val))
                self.all_eventraces_foracombo_eventcomboname_dict[
                    key
                ], self.all_speedlists_foracombo_eventcomboname_dict[
                    key
                    ] = self.create_all_events_2d_lists(
                    list(val), self.start_choice_or_collect_times
                )
                # Before converting it to a df, we need to store this 2d array somewhere
                # and store the corresponding avg traces too
                """Perform some process on this dict you just updated"""
                # ??????Necessary???

                # converting that 2d list of lists for one subevent into df

                group_df_dff = pd.DataFrame.from_records(
                    self.all_eventraces_foracombo_eventcomboname_dict[key]
                )

                group_df_speed = pd.DataFrame.from_records(
                    self.all_speedlists_foracombo_eventcomboname_dict[key]
                )

                group_df_dff = self.trim_grouped_df(group_df_dff)

                group_df_speed = self.trim_grouped_df(group_df_speed)
                # print(group_df_dff.head())
                # Doing some editing on this df
                group_df_dff = Utilities.rename_all_col_names(group_df_dff, x_axis)

                group_df_speed = Utilities.rename_all_col_names(group_df_speed, x_axis)
                """print(
                    "Event %s has %s events omitted." % (str(key), self.events_omitted)
                )"""
                # print("Dimensions of grouped df:", (group_df_dff.shape))
                group_df_dff.insert(
                    loc=0,
                    column="Event #",
                    value=Utilities.make_value_list_for_col(
                        "Event", number_of_event_appearances - self.events_omitted
                    ),
                )

                group_df_speed.insert(
                    loc=0,
                    column="Event #",
                    value=Utilities.make_value_list_for_col(
                        "Event", number_of_event_appearances - self.events_omitted
                    ),
                )

                # print(group_df_dff)
                # print(type(key))
                # making a path for this df to go to (within session path)

                combo_name = str(key)

                new_path = os.path.join(
                    self.session_path,
                    "SingleCellAlignmentData",
                    self.cell_name,
                    self.get_event_traces_name(),
                    combo_name,
                )
                # Insert to aligned dff dict that corresponds to this object
                # self.aligned_dff_dict[self.get_event_traces_name] = group_df_dff

                os.makedirs(new_path, exist_ok=True)

                name_of_dff_df = "plot_ready.csv"
                csv_path_dff = os.path.join(new_path, name_of_dff_df)
                group_df_dff.to_csv(csv_path_dff, index=False)

                name_of_speed_df = "speed_plot_ready.csv"
                csv_path_speed = os.path.join(new_path, name_of_speed_df)
                group_df_speed.to_csv(csv_path_speed, index=False)

                ### Add on analysis here ###
                
                # 1) Averaging
                Utilities.avg_cell_eventrace(
                    csv_path_dff, self.cell_name, plot=True, export_avg=True
                )
                Utilities.avg_cell_event_dff_speed_norm(
                    csv_path_dff, csv_path_speed, self.cell_name, plot=True, export_avg=True
                )


                # make sure the events omitted resets after ever subcombo within an eventtrace
                self.events_omitted = 0
            else:
                print(f"WILL NOT INCLUDE {str(key)}")
