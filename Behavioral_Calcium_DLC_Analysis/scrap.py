"""PROGRAM FEATURE 1: Extracting/aligning data by parts (modularity).

When including just root folder path: <- Does all mice, sessions that can match, 
combos that can match, subcombos that can match.

d = {
    Root {
        Mouse 1: {
            Session 1: {
                Combo 1: {
                    Subcombo 1: {
                        concat_cells.csv
                    }
                }
            },
        }
    }
}
        
When including just mice paths: <- Does all sessions, combos, and subcombos that can match 
across X mice.

d = {
    Mouse 1: {
        Session 1: {
            Combo 1: {
                Subcombo 1: {
                    concat_cells.csv
                }
            }
        },
    }
}

When including just sessions paths. <- Does all combos and sub combos for the given sessions
you've indicated (which should be across different mice). A checker will check whether these 
sessions actually match.

d = {
    Session 1: {
        Combo 1: {
            Subcombo 1: {
                concat_cells.csv
            }
        }
    },
}

When including just combo paths <- Does all subcombos for the given combos you've indicated
(which should be across different mice). A checker will check whether the combo's sessions 
and combos themselves match.

d = {
    Combo 1: {
        Subcombo 1: {
            concat_cells.csv
        }
    }
}

When including just subcombo paths <- Does one across-mice-cell concatenation between two similar
events. Program will check whether the subcombo's combo, subcombo's sessions, and combos.

d = {
    Subcombo 1: {
        concat_cells.csv
    }
}

"""

""" PROGRAM FEATURE 2: Extracting/aligning subsets of data for the entire database.

What if you want to apply an entire process for all the information you have avaliable but
only require a subset of info to be extracted from it (instead of having to run everything
to get what you want)?

In this case, add conditionals to limit the data extraction extraction process.

"""
from typing import List
from pathlib import Path
import pprint
import re, os, glob
import json
import os.path as path


def walk(top, topdown=True, onerror=None, followlinks=False, maxdepth=None):
    islink, join, isdir = path.islink, path.join, path.isdir

    try:
        names = os.listdir(top)
    except OSError as err:
        if onerror is not None:
            onerror(err)
        return

    dirs, nondirs = [], []
    for name in names:
        if isdir(join(top, name)):
            dirs.append(name)
        else:
            nondirs.append(name)

    if topdown:
        yield top, dirs, nondirs

    if maxdepth is None or maxdepth > 1:
        for name in dirs:
            new_path = join(top, name)
            if followlinks or not islink(new_path):
                for x in walk(
                    new_path,
                    topdown,
                    onerror,
                    followlinks,
                    None if maxdepth is None else maxdepth - 1,
                ):
                    yield x
    if not topdown:
        yield top, dirs, nondirs


class BetweenMiceAligment:
    """
    Functionality goes here.
    """

    def __init__(self, align_all: bool, align_paths: bool):

        if align_all == True and align_paths == False:
            # Run entire database, don't consider any flexibility
            pass

        elif align_all == False and align_paths == True:
            # Don't run entire database, align from the level of specificity the user wants
            pass
        else:
            print("Not a valid input!")

    def check_if_string_in_list(self, input_str: str) -> bool:
        check = [i for i in self.avaliable_parameters_to_focus if (i == input_str)]
        return bool(check)

    def check_and_append(self, my_lst: List, input_str: str) -> None:
        if BetweenMiceAligment.check_if_string_in_list(input_str) == True:
            my_lst.append(input_str)
        else:
            print(f"{input_str} is not a valid parameter!")

    def check_if_root(self, key):
        if "root" in key:
            return True
        else:
            False

    def set_root_name(self, key, new_name):
        if BetweenMiceAligment.check_if_root(key) == True:
            self.data_hierarchy[new_name]
        else:
            pass

    def create_hierarchy_root(self):
        pass

    def find_mice_paths(self):
        pass

    def perform_big_process(self):
        pass

    def find_csv_paths(master_path, endswith) -> List:

        files = glob.glob(
            os.path.join(master_path, "**", "*%s") % (endswith),
            recursive=True,
        )

        return files

    def pretty(d, indent=0):
        for key, value in d.items():
            print("\t" * indent + str(key))
            if isinstance(value, dict):
                pprint.PrettyPrinter(value, indent + 1)
            else:
                print("\t" * (indent + 1) + str(value))

    def find_avg_dff_of_cell(session_path, endswith):

        files = glob.glob(
            os.path.join(session_path, "**", "*%s") % (endswith),
            recursive=True,
        )

        return files

    def concat_all_cells(lst_of_all_avg_concat_cells_path, root_path):
        pass


class Driver:

    """
    Connecting with the OS goes here.
    """

    # example 11/24/21: /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/Late Shock D1/BetweenCellAlignmentData/Block_Choice Time (s)/1.0/concat_cells.csv

    def main():
        """
        Root that contains mice data (doesn't have to be the direct root).
        """
        MASTER_PATH = Path(r"/media/rory/Padlock_DT/BLA_Analysis")

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
            "Shock Test",
            "Late Shock D1",
            "Late Shock D2",
        ]

        sessionname_sessionlist = {}
        session_count = 0

        for session_type in session_types:
            for root, dirs, files in walk(MASTER_PATH, maxdepth=4):
                if str(root).find(session_type) != -1:
                    # ^if a session type contains the keyword
                    session_path = root
                    if session_type in sessionname_sessionlist:
                        sessionname_sessionlist[session_type].append(session_path)
                        session_count += 1
                    else:
                        sessionname_sessionlist[session_type] = list()
                        sessionname_sessionlist[session_type].append(session_path)
                        session_count += 1

        # post-hoc dict editing, to not include some session where they not supposed to be
        for key in sessionname_sessionlist:
            if (
                str(key) == ("RDT D1")
                or str(key) == ("RDT D2")
                or str(key) == ("RDT D3")
            ):
                for count, ele in enumerate(sessionname_sessionlist[key]):
                    if str(ele).find("Post") != -1:  # post found
                        del sessionname_sessionlist[key][count]
                        session_count -= 1

            if str(key) == "RM D1":
                for count, ele in enumerate(sessionname_sessionlist[key]):
                    if str(ele).find("RM D10") != -1:  # post found
                        del sessionname_sessionlist[key][count]
                        session_count -= 1

        # Edit the session dict manually (THE CATEGORIZATION ALGO WONT BE PERFECT)
        sessionname_sessionlist = {
            "PR D1": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/PR D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-2/PR D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-3/PR D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/PR D1 NEW_SCOPE",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-7/PR D1 NEW_SCOPE",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-11/PR D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-13/PR D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-8/PR D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-9/PR D1",
            ],
            "PR D2": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/PR D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-2/PR D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-3/PR D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-5/PR D2 NEW_SCOPE",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-7/PR D2 NEW_SCOPE",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-11/PR D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-13/PR D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-8/PR D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-9/PR D2",
            ],
            "Pre-RDT RM": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/Pre-RDT RM",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-3/Pre-RDT RM",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/Pre-RDT RM",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-7/Pre-RDT RM",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-11/Pre-RDT RM",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-13/Pre-RDT RM",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-8/Pre-RDT RM",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-9/Pre-RDT RM",
            ],
            "RDT D1": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-3/RDT D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/RDT D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-7/RDT D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-11/RDT D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-13/RDT D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-8/RDT D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-9/RDT D1",
            ],
            "RDT D2": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-3/RDT D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-5/RDT D2 NEW_SCOPE",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/RDT D2 NEW_SCOPE",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-7/RDT D2 NEW_SCOPE",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-11/RDT D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-13/RDT D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-8/RDT D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-9/RDT D2",
            ],
            "RDT D3": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-5/RDT D3 NEW_SCOPE",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/RDT D3 NEW_SCOPE",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-7/RDT D3 NEW_SCOPE",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-11/RDT D3",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-13/RDT D3",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-8/RDT D3",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-9/RDT D3",
            ],
            "Post-RDT D1": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/Post-RDT D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-3/Post-RDT D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/Post-RDT D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-7/Post-RDT D1",
            ],
            "Post-RDT D2": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/Post-RDT D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-2/Post-RDT D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-3/Post-RDT D2",
            ],
            "Post-RDT D3": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/Post-RDT D3"
            ],
            "RM D1": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-5/RM D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/RM D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-7/RM D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-11/RM D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-13/RM D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-8/RM D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-9/RM D1",
            ],
            "RM D2": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-2/RM D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-5/RM D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/RM D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-7/RM D2",
            ],
            "RM D3": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/RM D3"
            ],
            "RM D8": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-5/RM D8 TANGLED",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-7/RM D8",
            ],
            "RM D9": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/RM D9"
            ],
            "RM D10": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RM D10"
            ],
            "Shock Test": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/Shock Test",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-2/Shock Test",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-3/Shock Test",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-5/Shock Test NEW_SCOPE",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/Shock Test NEW_SCOPE",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-7/Shock Test NEW_SCOPE",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-11/Shock Test",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-13/Shock Test",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-8/Shock Test",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-9/Shock Test",
            ],
            "Late Shock D1": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/Late Shock D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-3/Late Shock D1",
            ],
            "Late Shock D2": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/Late Shock D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-2/Late Shock D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-3/Late Shock D2",
            ],
        }

        # print(session_count)

        # json_obj = json.dumps(sessionname_sessionlist, indent=4)
        with open(os.path.join(MASTER_PATH, "mouse_concats.txt"), "w") as outfile:
            json.dump(sessionname_sessionlist, outfile, ensure_ascii=False, indent=4)

        """For each session found similar, for each of the combos in the session,
        for each subcombo in the combo, find the concat_cells.csv, open it, for each column 
        (a cell's avg dff trace), insert the column name and list of df traces to new dict for each subcombo,
        and put it in a folder in that same hierarchy."""

        # example path:
        # /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-3/RDT D1/BetweenCellAlignmentData/Block_Choice Time (s)/1.0/concat_cells.csv
        betweenmice_alignment_path = os.path.join(
            MASTER_PATH, "BetweenMiceAlignmentData"
        )

        for session_type in sessionname_sessionlist:
            for session_path in sessionname_sessionlist[session_type]:
                avg_cells_csv_paths = BetweenMiceAligment.find_avg_dff_of_cell(
                    session_path, "concat_cells.csv"
                )
                for concat_csv_path in avg_cells_csv_paths:
                    combo_type = concat_csv_path.split("/")[9]
                    subcombo_type = concat_csv_path.split("/")[10]
                    new_concat_path = os.path.join(
                        betweenmice_alignment_path,
                        session_type,
                        combo_type,
                        subcombo_type,
                    )

                    os.makedirs(new_concat_path, exist_ok=True)

    def main2():
        """
        Root that contains mice data (doesn't have to be the direct root).
        """
        MASTER_PATH = Path(r"/media/rory/Padlock_DT/BLA_Analysis")

        # Edit the session dict manually (THE CATEGORIZATION ALGO WONT BE PERFECT)
        sessionname_sessionlist = {
            "PR D1": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/PR D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-2/PR D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-3/PR D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/PR D1 NEW_SCOPE",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-7/PR D1 NEW_SCOPE",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-11/PR D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-13/PR D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-8/PR D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-9/PR D1",
            ],
            "PR D2": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/PR D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-2/PR D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-3/PR D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-5/PR D2 NEW_SCOPE",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-7/PR D2 NEW_SCOPE",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-11/PR D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-13/PR D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-8/PR D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-9/PR D2",
            ],
            "Pre-RDT RM": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/Pre-RDT RM",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-3/Pre-RDT RM",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/Pre-RDT RM",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-7/Pre-RDT RM",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-11/Pre-RDT RM",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-13/Pre-RDT RM",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-8/Pre-RDT RM",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-9/Pre-RDT RM",
            ],
            "RDT D1": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-3/RDT D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/RDT D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-7/RDT D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-11/RDT D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-13/RDT D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-8/RDT D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-9/RDT D1",
            ],
            "RDT D2": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-3/RDT D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-5/RDT D2 NEW_SCOPE",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/RDT D2 NEW_SCOPE",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-7/RDT D2 NEW_SCOPE",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-11/RDT D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-13/RDT D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-8/RDT D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-9/RDT D2",
            ],
            "RDT D3": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-5/RDT D3 NEW_SCOPE",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/RDT D3 NEW_SCOPE",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-7/RDT D3 NEW_SCOPE",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-11/RDT D3",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-13/RDT D3",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-8/RDT D3",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-9/RDT D3",
            ],
            "Post-RDT D1": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/Post-RDT D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-3/Post-RDT D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/Post-RDT D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-7/Post-RDT D1",
            ],
            "Post-RDT D2": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/Post-RDT D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-2/Post-RDT D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-3/Post-RDT D2",
            ],
            "Post-RDT D3": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/Post-RDT D3"
            ],
            "RM D1": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-5/RM D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/RM D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-7/RM D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-11/RM D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-13/RM D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-8/RM D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-9/RM D1",
            ],
            "RM D2": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-2/RM D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-5/RM D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/RM D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-7/RM D2",
            ],
            "RM D3": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/RM D3"
            ],
            "RM D8": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-5/RM D8 TANGLED",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-7/RM D8",
            ],
            "RM D9": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/RM D9"
            ],
            "RM D10": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RM D10"
            ],
            "Shock Test": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/Shock Test",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-2/Shock Test",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-3/Shock Test",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-5/Shock Test NEW_SCOPE",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/Shock Test NEW_SCOPE",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-7/Shock Test NEW_SCOPE",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-11/Shock Test",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-13/Shock Test",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-8/Shock Test",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-9/Shock Test",
            ],
            "Late Shock D1": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/Late Shock D1",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-3/Late Shock D1",
            ],
            "Late Shock D2": [
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/Late Shock D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-2/Late Shock D2",
                "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-3/Late Shock D2",
            ],
        }

        """For each session found similar, for each of the combos in the session,
        for each subcombo in the combo, find the concat_cells.csv, open it, for each column 
        (a cell's avg dff trace), insert the column name and list of df traces to new dict for each subcombo,
        and put it in a folder in that same hierarchy."""

        # example path:
        # /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-2/RM D2/BetweenCellAlignmentData/Block_Choice Time (s)/1.0/concat_cells.csv
        # /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/RDT D1/BetweenCellAlignmentData/Block_Choice Time (s)/1.0/concat_cells.csv
        # /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/Post-RDT D2/SingleCellAlignmentData/BetweenCellAlignmentData/Block_Choice Time (s)/1.0/concat_cells.csv
        betweenmice_alignment_path = os.path.join(
            MASTER_PATH, "BetweenMiceAlignmentData"
        )

        # MASTER_DICT = {}

        for session_type in sessionname_sessionlist:
            # MASTER_DICT[session_type] = {}
            for session_path in sessionname_sessionlist[session_type]:
                avg_cells_csv_paths = BetweenMiceAligment.find_avg_dff_of_cell(
                    session_path, "concat_cells.csv"
                )
                for concat_csv_path in avg_cells_csv_paths:
                    if concat_csv_path.find("SingleCellAlignmentData") == -1:
                        combo_type = concat_csv_path.split("/")[9]
                        subcombo_type = concat_csv_path.split("/")[10]

                        new_concat_path = os.path.join(
                            betweenmice_alignment_path,
                            session_type,
                            combo_type,
                            subcombo_type,
                        )

                        os.makedirs(new_concat_path, exist_ok=True)
                    elif (
                        "SingleCellAligmentData"
                        and "BetweenCellAlignmentData" in concat_csv_path
                    ):  # string is found
                        del_path = concat_csv_path.split("/")[0:10]
                        del_path = "/".join(del_path)
                        print(del_path)


Driver.main2()
# Cleaner.main()

"""def __init__(
    self,
    root_path: str,
    mice: List,
    sessions_tofind: List,
    combos_tofind: List,
    avg_cell_traces_filename,
):
    self.root_path = root_path
    self.mice = mice
    self.sessions_tofind = sessions_tofind
    self.combos_tofind = combos_tofind
    self.avg_cell_traces_filename = avg_cell_traces_filename"""

"""def __init__(self, avg_cell_traces_filename, **kwargs):
    #Only sublevel processing argument names you can include.
    allowed_parameters = ["root", "mice", "sessions", "combos", "subcombos"]
    self.avg_cell_traces_filename = avg_cell_traces_filename
    self.data_hierarchy = (
        {}
    )  # Very important, dictates how folder/file structure will be

    for key, val in kwargs.items():
        valid_param = [i for i in allowed_parameters if (i in key)]
        if bool(valid_param) == True:
            self.key = key
            BetweenMiceAligment.set_root_name(key, new_name="BetweenMiceAlignment")
            # ^if not root, skip this making it in dict

    # ^the check that allows parameters to be used"""
"""
    elif align_all == False and align_all_subset == True and align_paths == False:
        # Run entire database, flexibility on what to acquire

        params = input(
            f"Choose which parameters you care about: {' '.join(self.avaliable_parameters_to_focus)}"
            + "\nType 'done' to indicate your done."
        )

        done = None
        params_to_focus = ()  # order matters, so a tuple
        while done != "done":
            new_input = input()
            params_to_focus.check_and_append(params_to_focus, new_input)

        for i in params_to_focus:
            specified_params = input(f"Which {i} are you focused on?")
"""
