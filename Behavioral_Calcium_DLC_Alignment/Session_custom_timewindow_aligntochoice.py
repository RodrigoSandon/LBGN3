import pandas as pd
import numpy as np
from pathlib import Path
import os
import glob
import Utilities_v2 as Utilities
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
        self.final_time_len = 0
        self.end = end
        self.groupby_list = groupby_list
        self.events_omitted = 0
        self.reference_start_choice_len = None
        self.reference_choice_end_len = None
        self.longest_sequence = None
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
        largest_start_to_choice_array = []
        reference_array = [] # copy of unedited largest_start_to_choice_array
        # bc largest_start_to_choice_array can get edited if either len (start - choice & choice - end) 
        # is shorter than another array even tho that array is shorter
        # if another array is equal, nothing happens
        reference_start = None
        reference_choice_time = None
        reference_choice_time_idx = None
        reference_end = None
        reference_start_choice_len = 0
        reference_choice_end_len = 0
        
        
        # FIRST FIGURE OUT LONGEST INDEX AND SAVE THOSE INDICIES
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
                str(start_time) != "nan" and str(collection_time) != "nan" and str(choice_time) != "nan"
            ):  # if the time is nan, then we don't include it in the stack of dff traces

                idx_df_lower_bound_time, lower_time_val = self.find_idx(
                    start_time
                )
                
                idx_df_upper_bound_time, upper_time_val = self.find_idx(
                    collection_time
                )
                choice_time_idx, choice_time_val = self.find_idx(choice_time)
                
                dff_trace = list(
                        self.get_dff_traces_of_neuron()[
                            idx_df_lower_bound_time:idx_df_upper_bound_time
                        ][self.cell_name]
                    )

                curr_start_to_choice = choice_time_idx - idx_df_lower_bound_time
                curr_choice_to_end = idx_df_upper_bound_time - choice_time_idx

                if curr_start_to_choice > reference_start_choice_len:
                    reference_start_choice_len = curr_start_to_choice
                    
                    #print(idx_df_lower_bound_time)
                    #print(idx_df_upper_bound_time)
                    largest_start_to_choice_array = dff_trace
                    reference_array = dff_trace
                    largest_start_to_choice_array_idx = abet_idx
                    reference_start = idx_df_lower_bound_time
                    #print(reference_start)
                    reference_choice_time = choice_time_idx
                    reference_end = idx_df_upper_bound_time
                    

                if curr_choice_to_end > reference_choice_end_len:
                    reference_choice_end_len = curr_choice_to_end
        #print(reference_end)
        #print(reference_start)

        #if reference_end != None and reference_start != None and reference_choice_time != None:
        #print(reference_choice_time, reference_start)
        reference_start_choice_len = reference_choice_time - reference_start
        self.reference_start_choice_len = reference_choice_time - reference_start # same variable but global
        reference_choice_end_len = reference_end - reference_choice_time
        self.reference_choice_end_len = reference_end - reference_choice_time # same variable but global
        reference_total_len = reference_end - reference_start
        self.reference_choice_time_idx = reference_choice_time
        
        # Now you have largest length of index
        # But now there's the issue of identifying this array, let's identify it by it's dff trace beforehand
        # Get the choice time idx
        #print(f"Largest array size : {len(largest_start_to_choice_array)}")
        #print(f"Reference start to choice len: {reference_start_choice_len}")
        # now identified largest array, matter of adjusting all other arrays based on this one
        # Now have choice time index for largest one, now adjust other trial arrays accordingly depending on their length
        # 
        event = 1
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
               
                """print(
                    (
                        "desired START time to extract dff traces from: %s, actual time extracted: %s"
                    )
                    % (start_time, lower_time_val)
                )"""
                idx_df_upper_bound_tim, upper_time_val = self.find_idx(
                    collection_time
                )
                
                """print(
                    (
                        "desired END time to extract dff traces from: %s, actual time extracted: %s"
                    )
                    % (collection_time, upper_time_val)
                )"""
                choice_time_idx, choice_time_val = self.find_idx(choice_time)
                #print(f"Choice time idx @ {event}: {choice_time_idx}")

                dff_block_of_neuron = list(
                    self.get_dff_traces_of_neuron()[
                        idx_df_lower_bound_time:idx_df_upper_bound_tim
                    ][self.cell_name]
                )
                #print(len(dff_block_of_neuron))
                ############### MODIFYING THESE TRACES ###############
                # if either len of reference is equal to len of curr, then don't do anything

                if dff_block_of_neuron != reference_array:
                    curr_start_choice_len = choice_time_idx - idx_df_lower_bound_time
                    #print(curr_start_choice_len)
                    curr_choice_end_len = idx_df_upper_bound_tim - choice_time_idx

                    
                    how_many_fillers_to_add_start = reference_start_choice_len - curr_start_choice_len
                    how_many_fillers_to_add_end = reference_choice_end_len - curr_choice_end_len

                    temp_list_start = [np.nan for i in range(how_many_fillers_to_add_start)]
                    temp_list_end = [np.nan for i in range(how_many_fillers_to_add_end)]
                    dff_block_of_neuron = temp_list_start + dff_block_of_neuron + temp_list_end

                ############### MODIFYING THESE TRACES ###############
                # print(dff_block_of_neuron) - only one dff block, so something wrong upstream
                    list_of_lists.append(dff_block_of_neuron)

                # only getting the dff, not considering the relative time, just absolute time
                # now append this
            else:
                self.events_omitted += 1
                pass
            event += 1

        list_of_lists.append(largest_start_to_choice_array)
        longest = 0
        for i in list_of_lists:
            if len(i) > longest:
                longest = len(i)
        self.longest_sequence = longest
        self.final_time_len = len(largest_start_to_choice_array)
        return list_of_lists  # this is a 2d list - SCOPE OF THIS WAS INNER

    def get_xaxis_list_for_plotting(self):
        
        #print(f"final start to choice time len: {self.reference_start_choice_len}")
        #print(f"final choice to end time len: {self.reference_choice_end_len}")
        # However,self.reference_choice_end_len + necesarily the longest + start to choice isn't
        # necesarily the longest sequence we'll get,
        #print(self.longest_sequence)
        t_neg = np.arange(-(self.reference_start_choice_len/10), 0.0, 0.1)
        t_pos = np.arange(0.0, ((self.longest_sequence - self.reference_start_choice_len)/10), 0.1)
        
        t = t_neg.tolist() + t_pos.tolist()
        t = [round(i, 1) for i in t]
        
        t_alt = np.arange(0, self.final_time_len, 0.1).tolist()
        #print(t)
        #print(len(t))
        t = t[:self.reference_start_choice_len] + ["Choice Time"] + t[self.reference_start_choice_len + 1:]
        #print(t)
        #print(len(t))

        return t

    def process_dff_traces_by(self):

        grouped_table = self.get_abet().groupby(self.groupby_list)
        
        # SUBCOMBO PROCESSING
        for key, val in grouped_table.groups.items():
            # make sure to not include subcombos that have nans in it

            if "nan" not in str(key) and "ITI" not in str(key) and "Omit" not in str(key):
                #omit iti bc they have no choice time

                number_of_event_appearances = len(list(val))
                print(key, ": ", list(val))

                self.alleventracesforacombo_eventcomboname_dict[
                    key
                ] = self.stack_dff_traces_of_group(
                    list(val), self.start, self.end
                )

                x_axis = self.get_xaxis_list_for_plotting()
                #print(x_axis)
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
                name_of_csv = "plot_ready_choice_aligned.csv"
                csv_path_aligned = os.path.join(new_path, name_of_csv)
                group_df.to_csv(csv_path_aligned, index=False)

                ### Add on analysis here ###
                df = pd.read_csv(csv_path_aligned)
                #print(df.head())
                df = df.T
                df = df.iloc[1:, :]  # omit first col
                df = df.T
                
                # 1) Average
                Utilities.avg_cell_eventrace(
                    df, csv_path_aligned, self.cell_name, plot=True, export_avg=True
                )

                # 2) Resample then Average
                # first create _resampled.csv
                
                t_pos = np.arange(0.0, 10.1, 0.1)
                t_neg = np.arange(-10.0, 0.0, 0.1)
                t = t_neg.tolist() +  t_pos.tolist()
                t = [round(i, 1) for i in t]
                #print(len(t))
                Utilities.create_resampled_plot_ready(csv_path_aligned, t, desired_start_choice_len=100, desired_choice_end_len=100)
                csv_path_aligned_resampled = os.path.join(new_path, "plot_ready_choice_aligned_resampled.csv")

                new_df = pd.read_csv(csv_path_aligned_resampled)
                new_df = new_df.iloc[:, 1:]
                #print(new_df.head())
                cell_name = csv_path_aligned_resampled.split("/")[9]
                Utilities.avg_cell_eventrace_w_resampling(new_df, csv_path_aligned_resampled, cell_name, t, plot=True, export_avg=True)

                # make sure the events omitted resets after ever subcombo within an eventtrace
                self.events_omitted = 0
            else:
                print("WILL NOT INCLUDE %s" % (str(key)))