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
        Session.session_path = session_path
        Session.session_id = session_path.split("/")[6]
        # loading options: "dff" "dlc" "abet"
        Session.dff_traces = self.load_table("dff")
        Session.dlc_df = self.load_table("sleap")
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
            if cell_name != "Time(s)":
                sub_dict_for_neuron = {
                    "Time(s)": d["Time(s)"], cell_name: dff_traces}
                sub_df_for_neuron = pd.DataFrame.from_dict(sub_dict_for_neuron)
                neuron_obj = Neuron(cell_name, sub_df_for_neuron)
                neurons[cell_name] = neuron_obj

        return neurons

    """ Loading DFF, DLC, and ABET tables onto session"""

    def load_table(self, table_to_extract):
        if table_to_extract == "dff":
            path = Utilities.find_dff_trace_path(
                self.session_path, "dff_traces_preprocessed.csv"
            )
            if path == None:
                print("No dff table found!")
                return None
            dff_traces = pd.read_csv(path)
            return dff_traces
        elif table_to_extract == "sleap":
            path = Utilities.find_dff_trace_path(
                self.session_path, "_sleap_data.csv"
            )
            if path == None:
                print("No SLEAP table found!")
                return None
            dlc_df = pd.read_csv(path)
            return dlc_df
        elif table_to_extract == "abet":
            path = Utilities.find_dff_trace_path(
                self.session_path, "_ABET_GPIO_processed.csv"
            )
            if path == None:
                print("No ABET table found!")
                return None
            behavioral_df = pd.read_csv(path)

            return behavioral_df
        else:
            print("Type a valid table to load!")
            return None

    def get_neurons(self) -> dict:
        return self.neurons

    def get_session_id(self):
        return self.id

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

    def __init__(self, cell_name, dff_trace_df):
        self.cell_name = cell_name
        self.dff_trace = dff_trace_df

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
        start,
        end,
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
                    + start + "_" + end
                )
                self.categorized_dff_traces[event_name] = list(
                    combine_by_list)  # can be omitted

                self.categorized_dff_traces[event_name] = EventTrace(
                    self.cell_name,
                    self.dff_trace,
                    event_name,
                    start,
                    end,
                    list(combine_by_list),
                )


class EventTrace(Neuron):  # for one combo

    alleventracesforacombo_eventcomboname_dict = {}  # values are in 2d


    avgeventracesforacombo_eventcomboname_dict = {}  # values are in 1d


    def __init__(
        self,
        cell_name,
        dff_trace,
        eventtraces_name,
        start,
        end,
        groupby_list: list,
    ):
        # so this EventTraces obj should have a name
        self.name = eventtraces_name
        self.start = start
        self.final_start_val = 999999
        self.end = end
        self.final_end_val = -999999
        self.groupby_list = groupby_list
        self.events_omitted = 0
        super().__init__(cell_name, dff_trace)

    def get_event_traces_name(self):
        return self.name

    def get_dff_traces_of_neuron(self):
        return self.get_dff_trace()

    def get_abet(self):
        return self.behavioral_df  # the first row has been deleted

    """TODO: Add another parameter to this function, the dlc data, eventually"""

    def find_idx(self, time): # gets closest time val to input + index
        # get col of time as a list for the df of neuron
        idx = Utilities.binary_search(
            list(self.get_dff_traces_of_neuron()["Time(s)"]), time
        )
        idx_time_val = self.get_dff_traces_of_neuron().iloc[
            idx, self.get_dff_traces_of_neuron().columns.get_loc("Time(s)")
        ]

        return idx, idx_time_val

    def stack_dff_traces_of_group(self, list_of_idxs, start, end, choice_col_name="Choice Time (s)"):
        """This is for one grouping found from the groupby columns"""
        
        list_of_lists = []

        for abet_idx in list_of_idxs:
            start_time = self.get_abet().iloc[
                abet_idx, self.get_abet().columns.get_loc(start)
            ]
            collection_time = self.get_abet().iloc[
                abet_idx, self.get_abet().columns.get_loc(end)
            ]

            choice_time = self.get_abet().iloc[
                abet_idx, self.get_abet().columns.get_loc(choice_col_name)
            ]
            if (
                str(start_time) != "nan" and str(collection_time) != "nan" 
            ):  # if the time is nan, then we don't include it in the stack of dff traces

                idx_df_lower_bound_time, lower_time_val = self.find_idx(
                    start_time
                )
                if lower_time_val < self.final_start_val:
                    self.final_start_val = lower_time_val
                print(
                    (
                        "desired START time to extract dff traces from: %s, actual time extracted: %s"
                    )
                    % (start_time, lower_time_val)
                )
                idx_df_upper_bound_tim, upper_time_val = self.find_idx(
                    collection_time
                )
                if upper_time_val > self.final_end_val:
                    self.final_end_val = upper_time_val
                print(
                    (
                        "desired END time to extract dff traces from: %s, actual time extracted: %s"
                    )
                    % (collection_time, upper_time_val)
                )
                choice_time_idx, choice_time_val = self.find_idx(choice_time)
                dff_block_of_neuron = list(
                    self.get_dff_traces_of_neuron()[
                        idx_df_lower_bound_time:idx_df_upper_bound_tim
                    ][self.cell_name]
                )
                
                # print(dff_block_of_neuron) - only one dff block, so something wrong upstream
                list_of_lists.append(dff_block_of_neuron)
                # only getting the dff, not considering the relative time, just absolute time
                # now append this
            else:
                self.events_omitted += 1
                pass
        
        return list_of_lists  # this is a 2d list - SCOPE OF THIS WAS INNER

    def get_xaxis_list_for_plotting(self):
        """Hertz of frames is 10 Hz, so increment by 0.1 within this time window.

        Returns: path to where csv was saved."""
        #print(self.start_val)
        #print(self.end_val)
        return np.arange(
            self.final_start_val, self.final_end_val, 0.1
        ).tolist()

    def process_dff_traces_by(self):

        grouped_table = self.get_abet().groupby(self.groupby_list)
        x_axis = self.get_xaxis_list_for_plotting()
        # SUBCOMBO PROCESSING
        for key, val in grouped_table.groups.items():
            # make sure to not include subcombos that have nans in it

            if "nan" not in str(key):

                number_of_event_appearances = len(list(val))
                # print(key, ": ", list(val))
                self.alleventracesforacombo_eventcomboname_dict[
                    key
                ] = self.stack_dff_traces_of_group(
                    list(val), self.start, self.end
                )
                
                # Before converting it to a df, we need to store this 2d array somewhere
                # and store the corresponding avg traces too
                """Perform some process on this dict you just updated"""
                # ??????Necessary???

                # converting that 2d list of lists into df

                group_df = pd.DataFrame.from_records(
                    self.alleventracesforacombo_eventcomboname_dict[key]
                )
                
                group_df = Utilities.rename_all_col_names(group_df, x_axis)
                
                group_df.insert(
                    loc=0,
                    column="Event #",
                    value=Utilities.make_value_list_for_col(
                        "Event", number_of_event_appearances - self.events_omitted
                    ),
                )

                combo_name = str(key)

                new_path = os.path.join(
                    self.session_path,
                    "SingleCellAlignmentData",
                    self.cell_name,
                    self.get_event_traces_name(),
                    combo_name,
                )
                # Insert to aligned dff dict that corresponds to this object
                # self.aligned_dff_dict[self.get_event_traces_name] = group_df

                os.makedirs(new_path, exist_ok=True)
                name_of_csv = "plot_ready.csv"
                csv_path = os.path.join(new_path, name_of_csv)
                group_df.to_csv(csv_path, index=False)

                ### Add on analysis here ###
                df = pd.read_csv(csv_path)
                #print(df.head())
                df = df.T
                df = df.iloc[1:, :]  # omit first col
                df = df.T
                
                # 2) Average Z score per each trial
                Utilities.avg_cell_eventrace(
                    df, csv_path, self.cell_name, plot=True, export_avg=True
                )
                # make sure the events omitted resets after ever subcombo within an eventtrace
                self.events_omitted = 0
            else:
                print("WILL NOT INCLUDE %s" % (str(key)))