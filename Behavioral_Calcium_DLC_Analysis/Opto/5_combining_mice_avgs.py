from typing import List
from pathlib import Path
import pprint
import re
import os
import glob
import json
import os.path as path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def find_csv_files(session_path, startswith):

    files = glob.glob(
        os.path.join(session_path, "**", "%s*") % (startswith),
        recursive=True,
    )

    return files

def add_val(arr, val):
    
    return np.append(arr, [val])

def plot_indv_speed(csv_path):
    """Plots the figure from the csv file given"""
    df = pd.read_csv(csv_path)
    #print(df.head())
    num_mice = len(list(df.columns)[1:])

    fig, ax = plt.subplots()
    every_nth = 30
    # add the last value

    t = list(df["Time_(s)"]) + [5.0]

    for col in list(df.columns)[1:]:
        y = add_val(np.array(list(df[col])), np.nan)
        #print(y)
        ax.plot(t, y)

    ax.set_xticks([round(i, 1) for i in t])
    ax.set_xlabel("Time from trigger (s)")
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    for n, label in enumerate(ax.xaxis.get_major_ticks()):
        if n % every_nth != 0:
            label.set_visible(False)
    ax.set_ylabel("Average speeds (cm/s)")
    ax.set_title(f"Avg. Speeds of Mice, Z-scored, Savitzky (n={num_mice})")
    fig.savefig(csv_path.replace(".csv",".png"))
    plt.close(fig)

def concat_all_cells_across_similar_sessions(
    lst_of_all_avg_concat_cells_path, root_path
):
    between_mice_d = {}
    """"
    d = 
    {
        session_type : {
            combo: {
                subcombo: {
                    mouse_session : [avg speed],
                    mouse_session : [avg speed],
                    mouse_session : [avg speed],

                }
            }
        }
    }
    """
    # /media/rory/RDT VIDS/BORIS/RRD170/RDT OPTO CHOICE 0115/AlignmentData/Block_Trial_Type_Reward_Size_Start_Time_(s)/(1.0, 'Forced', 'Large')/speeds_z_-5_5savgol_avg.csv
    for avg_concat_cells_csv_path in lst_of_all_avg_concat_cells_path:
        print(f"Currently working on ...{avg_concat_cells_csv_path}")
        mouse_name = avg_concat_cells_csv_path.split("/")[5]
        ###PARAMS CATEGORIZED BY###
        if "CHOICE" in avg_concat_cells_csv_path.split("/")[6]:

            session_type = "Choice"
        else:
            session_type = "Outcome"

        combo = avg_concat_cells_csv_path.split("/")[8]
        subcombo = avg_concat_cells_csv_path.split("/")[9]

        # means nothing has been created yet
        if session_type not in between_mice_d:
            between_mice_d[session_type] = {}
            between_mice_d[session_type][combo] = {}
            between_mice_d[session_type][combo][subcombo] = {}
            avg_dff_traces_df = pd.read_csv(avg_concat_cells_csv_path)
          
            for col_name, col_data in avg_dff_traces_df.iteritems():
                cell_name = col_name
                # print(cell_name)
                # for however many cells there are under this mouse, for this session type, combo, n subcombo
                between_mice_d[session_type][combo][subcombo][
                    cell_name
                ] = avg_dff_traces_df[cell_name].tolist()

        # if there is already a session type key
        elif session_type in between_mice_d:

            if combo not in between_mice_d[session_type]:
                between_mice_d[session_type][combo] = {}
                between_mice_d[session_type][combo][subcombo] = {}
                avg_dff_traces_df = pd.read_csv(avg_concat_cells_csv_path)
                
                for col_name, col_data in avg_dff_traces_df.iteritems():
                    
                    cell_name = col_name
                    between_mice_d[session_type][combo][subcombo][
                        cell_name
                    ] = avg_dff_traces_df[cell_name].tolist()

            elif combo in between_mice_d[session_type]:

                if subcombo not in between_mice_d[session_type][combo]:
                    between_mice_d[session_type][combo][subcombo] = {}
                    avg_dff_traces_df = pd.read_csv(avg_concat_cells_csv_path)
                    
                    for col_name, col_data in avg_dff_traces_df.iteritems():
                        cell_name = col_name
                        between_mice_d[session_type][combo][subcombo][
                            cell_name
                        ] = avg_dff_traces_df[cell_name].tolist()

                elif subcombo in between_mice_d[session_type][combo]:
                    # this is where you add on that foreign cell
                    avg_dff_traces_df = pd.read_csv(avg_concat_cells_csv_path)
                    
                    for col_name, col_data in avg_dff_traces_df.iteritems():
                        # only pull speeds, no more times
                        if col_name != f"{mouse_name}_Time_(s)":
                            cell_name = col_name
                            # print(cell_name)
                            between_mice_d[session_type][combo][subcombo][
                                cell_name
                            ] = avg_dff_traces_df[cell_name].tolist()

    for session_type in between_mice_d:
        for combo in between_mice_d[session_type]:
            for subcombo in between_mice_d[session_type][combo]:
                # now have all cells for combo, now just combine them all to one csv
                # print(event_dict[event][combo])
                try:
                    concatenated_cells_df = pd.DataFrame.from_dict(
                        between_mice_d[session_type][combo][subcombo]
                    )
                except ValueError:
                    # print(between_mice_d[session_type][combo][subcombo])
                    print("JAGGED ARRAYS IN:", session_type, combo, subcombo)
                    d = between_mice_d[session_type][combo][subcombo]
                    concatenated_cells_df = pd.DataFrame(
                        dict([(k, pd.Series(v)) for k, v in d.items()])
                    )

                new_path = os.path.join(root_path, session_type, combo, subcombo)
                os.makedirs(new_path, exist_ok=True)
                concatenated_cells_df.to_csv(
                    os.path.join(new_path, "all_speeds_z_-5_5_savgol_avg.csv"), index=False
                )
                plot_indv_speed(os.path.join(new_path, "all_speeds_z_-5_5_savgol_avg.csv"))


def main():
    ROOT_PATH = Path(r"/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis_2/")
    
    def find_paths(root_path: Path, endswith: str) -> List[str]:
        files = glob.glob(
            os.path.join(root_path, "**", endswith), recursive=True,
        )
        return files

    lst_of_avg_cell_csv_paths = find_paths(ROOT_PATH, "speeds_z_-5_5_savgol_avg.csv") #CUSTOMIZE FOR SPECIFIC GROUPINGS YOU WANT TO PROCESS
    bw_mice_alignment_f_name = "BetweenMiceAlignmentData"
    bw_mice_alignment_path = os.path.join(ROOT_PATH, bw_mice_alignment_f_name)

    concat_all_cells_across_similar_sessions(
        lst_of_avg_cell_csv_paths, bw_mice_alignment_path
        )


main()