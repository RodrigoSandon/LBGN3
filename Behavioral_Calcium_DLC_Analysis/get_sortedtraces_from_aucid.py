import os, glob
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
from csv import writer
import math

def find_paths(root_path: Path, middle: str, endswith: str) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, "**", middle, "**", endswith), recursive=True,
    )
    return files

def find_paths_conditional_endswith(
    root_path, og_lookfor: str, cond_lookfor: str
) -> list:

    all_files = []

    for root, dirs, files in os.walk(root_path):

        if cond_lookfor in files:
            # acquire the trunc file
            file_path = os.path.join(root, cond_lookfor)
            # print(file_path)
            all_files.append(file_path)
        elif cond_lookfor not in files:
            # acquire the og lookfor
            file_path = os.path.join(root, og_lookfor)
            all_files.append(file_path)

    return all_files


class Cell:
    def __init__(
        self,
        dff_trace: list,
        number_cells,
        base_lower_bound_time,
        base_upper_bound_time,
        lower_bound_time,
        upper_bound_time,
        reference_pair,
        hertz,
        alpha,
    ):

        self.number_cells = number_cells
        self.base_lower_bound_time = base_lower_bound_time
        self.base_upper_bound_time = base_upper_bound_time
        self.lower_bound_time = lower_bound_time
        self.upper_bound_time = upper_bound_time
        self.reference_pair = reference_pair
        self.hertz = hertz
        self.alpha = alpha

        self.dff_trace = dff_trace
        self.id = self.wilcoxon_analysis()

    def convert_secs_to_idx(
        self, unknown_time_min, unknown_time_max, reference_pair: dict, hertz: int
    ):
        reference_time = list(reference_pair.keys())[0]  # has to come from 0
        reference_idx = list(reference_pair.values())[0]

        idx_start = (unknown_time_min * hertz) + reference_idx

        idx_end = (unknown_time_max * hertz) + reference_idx  # exclusive
        return int(idx_start), int(idx_end)

    def create_subwindow_of_list(
        self, lst: list, unknown_time_min, unknown_time_max
    ) -> list:
        idx_start, idx_end = self.convert_secs_to_idx(
            unknown_time_min, unknown_time_max, self.reference_pair, self.hertz
        )

        subwindow_lst = lst[idx_start:idx_end]
        return subwindow_lst

    def wilcoxon_analysis(self):

        sub_df_baseline_lst = self.create_subwindow_of_list(
            self.dff_trace,
            unknown_time_min=self.base_lower_bound_time,
            unknown_time_max=self.base_upper_bound_time,
        )

        sub_df_lst = self.create_subwindow_of_list(
            self.dff_trace,
            unknown_time_min=self.lower_bound_time,
            unknown_time_max=self.upper_bound_time,
        )

        if (sub_df_baseline_lst == sub_df_lst) == True:
            return "null"
        try:
            result_greater = stats.mannwhitneyu(
                sub_df_lst, sub_df_baseline_lst, alternative="greater"
            )

            result_less = stats.mannwhitneyu(
                sub_df_lst, sub_df_baseline_lst, alternative="less"
            )
        except ValueError:
            print(sub_df_baseline_lst)
            print(sub_df_lst)

        id = None
        if result_greater.pvalue < (self.alpha / self.number_cells):
            id = "+"
        elif result_less.pvalue < (self.alpha / self.number_cells):
            id = "-"
        else:
            id = "Neutral"

        return id

    def zscore(self, obs_value, mu, sigma):
        return (obs_value - mu) / sigma

    def custom_standardize(
        self, dff_traces: list, norm_base_min, norm_base_max
    ) -> list:

        subwindow = self.create_subwindow_of_list(
            dff_traces, norm_base_min, norm_base_max
        )
        mean_for_cell = stats.tmean(subwindow)
        stdev_for_cell = stats.tstd(subwindow)

        new_dff_traces = []
        for ele in dff_traces:
            z_score = self.zscore(ele, mean_for_cell, stdev_for_cell)
            new_dff_traces.append(z_score)

        return new_dff_traces


def main():
    ROOT = r"/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData"
    #dff_csv = "all_concat_cells_fullwindow_z_auc.csv"
    id_csv = "all_concat_cells_id_fullwindow_z_auc.csv"
    #if_folder_includes_this_process_this_instead = "all_concat_cells_z_pre_truncated.csv"

    """files = find_paths_conditional_endswith(
        ROOT, to_look_for, if_folder_includes_this_process_this_instead
    )"""

    list_of_sessions = ["RDT D1", "RDT D2", "RDT D3"]
    
    for i in list_of_sessions:
        print(i)
        files = find_paths(ROOT, i, id_csv)

        
        for id_csv in files:
            dff_csv = id_csv.replace("_id","")
            print(f"CURR ID CSV: {id_csv}")
            print(f"CORRESPONDING DFF CSV: {dff_csv}")
            try:

                #try:
                id_df = pd.read_csv(id_csv)
                dff_df = pd.read_csv(dff_csv)

                number_cells = len(list(id_df.columns))

                out_path = "/".join(id_csv.split("/")[:-1]) + "/sorted_traces_z_fullwindow_aucid.png"
                
                # print(out_path)

                # dict of lists of lists of dff_traces, would later be avg dff traces
                d = {"+": [], "-": [], "Neutral": []}
                d_description = {
                    "+": {
                        "mean":[],
                        "sem":[]
                    }, 
                    "-": {
                        "mean":[],
                        "sem":[]
                    }, 
                    "Neutral": {
                        "mean":[],
                        "sem":[]
                        }
                }

                # loop through this csv
                for count, col in enumerate(list(id_df.columns)):
                    
                    if list(id_df[col])[0] == "+":
                        d["+"].append(list(dff_df[col]))
                    elif list(id_df[col])[0] == "-":
                        d["-"].append(list(dff_df[col]))
                    else:
                        d["Neutral"].append(list(dff_df[col]))

                # Now have all cells sorted, find avg of lists of lists
                for key in d.keys():
                    
                    num_traces_in_list = len(d[key])
                    zipped_dff_traces = zip(*d[key])
                    # d[key] is a list of lists

                    avg_dff_traces = []

                    for index, tuple in enumerate(zipped_dff_traces):
                        
                        avg = sum(list(tuple)) / num_traces_in_list
                        sem = stats.tstd(list(tuple))/(math.sqrt(num_traces_in_list))
                        avg_dff_traces.append(avg)

                        d_description[key]["mean"].append(avg)
                        d_description[key]["sem"].append(sem)
                        

                    d[key] = avg_dff_traces

                out_path_csv = out_path.replace("sorted_traces_z_fullwindow_aucid.png", "sorted_traces_z_fullwindow_aucid.csv")

                new_d = {}
                for key_1 in d_description:
                    for key_2 in d_description[key_1]:
                        new_key = f"{key_1}_{key_2}"
                        new_d[new_key] = d_description[key_1][key_2]

                d_description_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in new_d.items() ]))
                d_description_df.to_csv(out_path_csv, index=False)
                
                for key in d.keys():
                    plt.plot(list(dff_df.index), d_description[key]["mean"], label=key)
                    difference_1 = [d_description[key]["mean"][idx] - d_description[key]["sem"][idx] for idx,i in enumerate(d_description[key]["mean"])]
                    difference_2 = [d_description[key]["mean"][idx] + d_description[key]["sem"][idx] for idx,i in enumerate(d_description[key]["mean"])]
                    plt.fill_between(list(dff_df.index), difference_1, difference_2)

                plt.title(f"Averaged Z-Scores of AUC-Determined Cell Identities (n={number_cells})")
                plt.locator_params(axis="x", nbins=20)
                plt.xlabel("Time (s)")
                plt.ylabel("Z-Score")
                plt.legend()
                plt.savefig(out_path)
                plt.close()
                # break

            except Exception as e:
                print(e)
                pass


if __name__ == "__main__":
    main()