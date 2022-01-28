from typing import List
from pathlib import Path
import pprint
import re
import os
import glob
import json
import os.path as path
import pandas as pd


def find_csv_files(session_path, startswith):

    files = glob.glob(
        os.path.join(session_path, "**", "%s*") % (startswith),
        recursive=True,
    )

    return files


# example 11/24/21: /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/Late Shock D1/BetweenCellAlignmentData/Block_Choice Time (s)/1.0/concat_cells.csv
# /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-7/RDT D1/BetweenCellAlignmentData/Block_Omission_Choice Time (s)/(3.0, 'ITI')/concat_cells.csv
# /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-11/RDT D1/BetweenCellAlignmentData/Block_Choice Time (s)/1.0/concat_cells.csv


def concat_all_cells_across_similar_sessions(
    lst_of_all_avg_concat_cells_path, root_path
):
    between_mice_d = {}
    """"
    root_path + d = {
                        session_type : {
                            combo: {
                                subcombo: {
                                    cell : [avg dff traces],
                                    cell : [avg dff traces],
                                    cell : [avg dff traces]
                                }
                            }
                        }
    }
    """

    for avg_concat_cells_csv_path in lst_of_all_avg_concat_cells_path:
        print(f"Currently working on ...{avg_concat_cells_csv_path}")
        mouse_name = avg_concat_cells_csv_path.split("/")[6]
        ###PARAMS CATEGORIZED BY###
        session_type = avg_concat_cells_csv_path.split("/")[7]

        # accounting for some caveats
        if session_type == "RDT D2 NEW_SCOPE":
            session_type = "RDT D2"
        elif session_type == "RDT D3 NEW_SCOPE":
            session_type = "RDT D3"
        elif session_type == "RM D8 TANGLED":
            session_type = "RM D8"
        elif session_type == "Shock Test NEW_SCOPE":
            session_type = "Shock Test"

        combo = avg_concat_cells_csv_path.split("/")[9]
        subcombo = avg_concat_cells_csv_path.split("/")[10]

        # means nothing has been created yet
        if session_type not in between_mice_d:
            between_mice_d[session_type] = {}
            between_mice_d[session_type][combo] = {}
            between_mice_d[session_type][combo][subcombo] = {}
            avg_dff_traces_df = pd.read_csv(avg_concat_cells_csv_path)
            # I have to enter each column so I can modify its name to include the mouse_name
            for col_name, col_data in avg_dff_traces_df.iteritems():
                avg_dff_traces_df = avg_dff_traces_df.rename(
                    columns={col_name: "_".join([mouse_name, col_name])}
                )
            # insert new cell to subcombo dict indv
            # Changed cell names, now insert as cell
            # (no check for cell again bc ive added the mouse name to cell name)
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
                    avg_dff_traces_df = avg_dff_traces_df.rename(
                        columns={col_name: "_".join([mouse_name, col_name])}
                    )
                for col_name, col_data in avg_dff_traces_df.iteritems():
                    cell_name = col_name
                    print(cell_name)
                    between_mice_d[session_type][combo][subcombo][
                        cell_name
                    ] = avg_dff_traces_df[cell_name].tolist()

            elif combo in between_mice_d[session_type]:

                if subcombo not in between_mice_d[session_type][combo]:
                    between_mice_d[session_type][combo][subcombo] = {}
                    avg_dff_traces_df = pd.read_csv(avg_concat_cells_csv_path)
                    for col_name, col_data in avg_dff_traces_df.iteritems():
                        avg_dff_traces_df = avg_dff_traces_df.rename(
                            columns={col_name: "_".join([mouse_name, col_name])}
                        )
                    for col_name, col_data in avg_dff_traces_df.iteritems():
                        cell_name = col_name
                        print(cell_name)
                        between_mice_d[session_type][combo][subcombo][
                            cell_name
                        ] = avg_dff_traces_df[cell_name].tolist()

                elif subcombo in between_mice_d[session_type][combo]:
                    # this is where you add on that foreign cell
                    avg_dff_traces_df = pd.read_csv(avg_concat_cells_csv_path)
                    for col_name, col_data in avg_dff_traces_df.iteritems():
                        avg_dff_traces_df = avg_dff_traces_df.rename(
                            columns={col_name: "_".join([mouse_name, col_name])}
                        )
                    for col_name, col_data in avg_dff_traces_df.iteritems():
                        cell_name = col_name
                        # print(cell_name)
                        between_mice_d[session_type][combo][subcombo][
                            cell_name
                        ] = avg_dff_traces_df[cell_name].tolist()

    # now should have every cell in their proper category (across mice)
    """with open(
        os.path.join(root_path, "all_categorized_cell_avg_traces.txt"), "w+"
    ) as outfile:
        json.dump(between_mice_d, outfile, ensure_ascii=False, indent=4)"""

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
                    os.path.join(new_path, "all_concat_cells.csv"), index=False
                )


def main():
    ROOT_PATH = Path(r"/media/rory/Padlock_DT/BLA_Analysis")
    """
    SESSION TYPES: 
        "PR D1", <- will never appear as of 11/29/21
        "PR D2", <- will never appear as of 11/29/21
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
        "Shock Test", <- will never appear as of 11/29/21
        "Late Shock D1",
        "Late Shock D2",
    
    SO IN TOTAL, 15 FOLDERS SHOULD APPEAR 11/29/21
    """

    # file = open(f"{ROOT_PATH}/structure_of_between_mice_alignment.txt", "w+")

    lst_of_avg_cell_csv_paths = find_csv_files(ROOT_PATH, "concat_cells.csv")
    bw_mice_alignment_f_name = "BetweenMiceAlignmentData"
    bw_mice_alignment_path = os.path.join(ROOT_PATH, bw_mice_alignment_f_name)

    concat_all_cells_across_similar_sessions(
        lst_of_avg_cell_csv_paths, bw_mice_alignment_path
    )
    # file.write("\n".join(lst_of_avg_cell_csv_paths))
    # file.close()


main()
