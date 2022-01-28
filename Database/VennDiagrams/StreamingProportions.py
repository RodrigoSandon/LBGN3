import sqlite3
import matplotlib
import pandas as pd
import copy
import os

import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
from matplotlib_venn import venn3_circles

from pprint import pprint as pp
import circlify as circ


"""
Assumptions:

1. You'll get results from both post/pre activity databases.
"""


class StreamProportions:
    def __init__(
        self,
        df: pd.DataFrame,
        c,
        db: str,
        session: str,
        analysis: str,
        start_choice_collect: str,
        subevent_chain: list,
    ):
        self.df = df
        self.c = c
        self.db = db
        self.session = session
        self.analysis = analysis
        self.start_choice_collect = start_choice_collect
        self.subevent_chain = subevent_chain
        # self.stream_activity()

    def make_substr_of_col_name(self, param: str):

        if " " in param:
            param = param.replace(" ", "_")

        col_name_substr = f"{self.analysis}_{param}_{self.start_choice_collect}"

        return col_name_substr

    def create_venndiagram_dict(self):
        d_activity = {}
        """
        d_activity = {
            subevent : {
                +_cells : [cell_name, ...],
                         .
                         .
                         .
            },
            subevent : {
                -_cells : [cell_name, ...],
                         .
                         .
                         .
            },
            subevent : {
                N_cells : [cell_name, ...],
                         .
                         .
                         .
            },
        }
        """
        d_responsiveness = {}
        """
        d_responsiveness = {
            subevent : {
                resp_cells : [cell_name, ...],
                         .
                         .
                         .
            },
            subevent : {
                nonresp_cells : [cell_name, ...],
                         .
                         .
                         .
            },
        }
        """

        for param in self.subevent_chain:  # param = a subevent
            # make the colname that your going to look for for this
            subevent_substr = param
            # now search it in the df, get the full name of col
            full_subevent_name = None
            for col in list(self.df.columns):
                if subevent_substr in col:
                    full_subevent_name = col
            # have subevent name
            # pull the cell_name col values
            # now have full col name, now i can id values for this subevent
            d_activity[full_subevent_name] = {
                "+_cells": [],
                "-_cells": [],
                "N_cells": [],
            }

            d_responsiveness[full_subevent_name] = {
                "resp_cells": [],
                "nonresp_cells": [],
            }

            for row_idx in range(len(self.df)):
                # pull the cell_name
                cell_name = self.df.iloc[row_idx]["cell_name"]
                # pull the id for the subevent
                id = self.df.iloc[row_idx][full_subevent_name]
                # insert to according list
                if id == "Neutral":
                    d_activity[full_subevent_name]["N_cells"].append(cell_name)
                    d_responsiveness[full_subevent_name]["nonresp_cells"].append(
                        cell_name
                    )
                elif id == "+":
                    d_activity[full_subevent_name]["+_cells"].append(cell_name)
                    d_responsiveness[full_subevent_name]["resp_cells"].append(cell_name)
                elif id == "-":
                    d_activity[full_subevent_name]["-_cells"].append(cell_name)
                    d_responsiveness[full_subevent_name]["resp_cells"].append(cell_name)

        return d_activity, d_responsiveness

    def find_subcategories_within_list(
        self, mylist: list, resp_list: list, nonresp_list: list
    ):
        new_resp_list = []
        new_nonresp_list = []

        for i in mylist:
            if i in resp_list:
                new_resp_list.append(i)
            elif i in nonresp_list:
                new_nonresp_list.append(i)
            # does not make a difference if elif or else

        # print("mylist:", len(mylist))
        """Here i return what the new list should look like (essentially updating it for
        the next time we want to find subcategories in a category)
        """
        return new_resp_list, [len(new_resp_list), len(new_nonresp_list)]

    def find_subcategories_pos(
        self, curr: list, pos_list: list, neg_list: list, n_list: list
    ):
        new_pos_list = []
        new_neg_list = []
        new_n_list = []

        for i in curr:
            if i in pos_list:
                new_pos_list.append(i)
            elif i in neg_list:
                new_neg_list.append(i)
            elif i in n_list:
                new_n_list.append(i)

        # print("mylist:", len(mylist))
        """Here i return what the new list should look like (essentially updating it for
        the next time we want to find subcategories in a category)
        """
        return new_pos_list, [len(new_pos_list), len(new_neg_list)]

    def find_subcategories_neg(
        self, curr: list, pos_list: list, neg_list: list, n_list: list
    ):
        new_pos_list = []
        new_neg_list = []
        new_n_list = []

        for i in curr:
            if i in pos_list:
                new_pos_list.append(i)
            elif i in neg_list:
                new_neg_list.append(i)
            elif i in n_list:
                new_n_list.append(i)

        return new_neg_list, [len(new_pos_list), len(new_neg_list)]

    def find_subcategories_n(
        self, curr: list, pos_list: list, neg_list: list, n_list: list
    ):
        new_pos_list = []
        new_neg_list = []
        new_n_list = []

        for i in curr:
            if i in pos_list:
                new_pos_list.append(i)
            elif i in neg_list:
                new_neg_list.append(i)
            elif i in n_list:
                new_n_list.append(i)

        return new_n_list, [len(new_pos_list), len(new_neg_list)]

    def stream_responsiveness(self):
        cell_ids_activity, cell_ids_responsiveness = self.create_venndiagram_dict()

        # OVERALL PROPORTION CHANGE ALONG SUBEVENTS
        labels = []
        resp_count = []
        nonresp_count = []

        # TRACKING ONLY RESP CELLS ALONG SUBEVENTS
        # subevent_name : [Resp, non-resp #]
        # subevent_name_subevent_name : [Resp, non-resp #]
        # subevent_name_subevent_name_subevent_name : [Resp, non-resp #]
        curr_resp = None
        tracking_resp = {}

        # TRACKING ONLY NON-RESP????

        for count, subevent in enumerate(cell_ids_responsiveness):
            # subevent_short = subevent.split("_")[1]
            if count == 0:
                labels.append(subevent)
                tracking_resp[subevent] = []
            else:
                labels.append(labels[count - 1] + "_" + subevent)

            # OVERALL PROPORTION
            resp_count.append(len(cell_ids_responsiveness[subevent]["resp_cells"]))
            nonresp_count.append(
                len(cell_ids_responsiveness[subevent]["nonresp_cells"])
            )
            # TRACKING RESP
            if count == 0:
                curr_resp = cell_ids_responsiveness[subevent]["resp_cells"]
                tracking_resp[subevent].append(
                    len(cell_ids_responsiveness[subevent]["resp_cells"])
                )
                tracking_resp[subevent].append(
                    len(cell_ids_responsiveness[subevent]["nonresp_cells"])
                )

            else:

                # We have a list of previous resp cells already, which of these are now resp.nonresp?
                # return new proportion here
                (
                    new_curr,
                    tracking_resp[labels[count - 1] + "_" + subevent],
                ) = self.find_subcategories_within_list(
                    curr_resp,
                    cell_ids_responsiveness[subevent]["resp_cells"],
                    cell_ids_responsiveness[subevent]["nonresp_cells"],
                )

                # update curr_resp, bc one has existed already
                # grabbing curr's subevent resp cells based on previous resp cells
                curr_resp = new_curr

        # print("CURR RESP:", len(curr_resp))
        print("Labels:", labels)
        print("Resp count:", resp_count)
        print("Non-resp count:", nonresp_count)
        print("TRACKING RESP")
        print(tracking_resp)

        # now make lists for how the proportions changed along the chain
        resp_chain = []
        nonresp_chain = []
        for key in tracking_resp:
            resp_chain.append(tracking_resp[key][0])
            nonresp_chain.append(tracking_resp[key][1])

        return resp_count, nonresp_count, resp_chain, nonresp_chain

    def stream_activity(self):
        cell_ids_activity, cell_ids_responsiveness = self.create_venndiagram_dict()

        # OVERALL PROPORTION CHANGE ALONG SUBEVENTS
        labels = []
        pos_count = []
        neg_count = []
        n_count = []

        # TRACKING POS/NEG CELLS ALONG SUBEVENTS
        # subevent_name : [+,-, N #]
        # subevent_name_subevent_name : [+,-, N #]
        # subevent_name_subevent_name_subevent_name : [+,-, N #]
        curr_pos = None
        tracking_pos = {}

        curr_neg = None
        tracking_neg = {}

        curr_n = None
        tracking_n = {}

        for count, subevent in enumerate(cell_ids_activity):
            # subevent_short = subevent.split("_")[1]
            if count == 0:
                labels.append(subevent)
                tracking_pos[subevent] = []
                tracking_neg[subevent] = []
                tracking_n[subevent] = []
            else:
                labels.append(labels[count - 1] + "_" + subevent)

            # OVERALL PROPORTION
            pos_count.append(len(cell_ids_activity[subevent]["+_cells"]))
            neg_count.append(len(cell_ids_activity[subevent]["-_cells"]))
            n_count.append(len(cell_ids_activity[subevent]["N_cells"]))

            # TRACKING POS/NEG
            if count == 0:
                curr_pos = cell_ids_activity[subevent]["+_cells"]
                curr_neg = cell_ids_activity[subevent]["-_cells"]
                curr_n = cell_ids_activity[subevent]["N_cells"]

                tracking_pos[subevent].append(
                    len(cell_ids_activity[subevent]["+_cells"])
                )
                tracking_pos[subevent].append(
                    len(cell_ids_activity[subevent]["-_cells"])
                )
                tracking_pos[subevent].append(
                    len(cell_ids_activity[subevent]["N_cells"])
                )
                ###

                tracking_neg[subevent].append(
                    len(cell_ids_activity[subevent]["+_cells"])
                )
                tracking_neg[subevent].append(
                    len(cell_ids_activity[subevent]["-_cells"])
                )
                tracking_neg[subevent].append(
                    len(cell_ids_activity[subevent]["N_cells"])
                )
                ###

                tracking_n[subevent].append(len(cell_ids_activity[subevent]["+_cells"]))
                tracking_n[subevent].append(len(cell_ids_activity[subevent]["-_cells"]))
                tracking_n[subevent].append(len(cell_ids_activity[subevent]["N_cells"]))

            else:

                (
                    new_curr_pos,
                    tracking_pos[labels[count - 1] + "_" + subevent],
                ) = self.find_subcategories_pos(
                    curr_pos,
                    cell_ids_activity[subevent]["+_cells"],
                    cell_ids_activity[subevent]["-_cells"],
                    cell_ids_activity[subevent]["N_cells"],
                )

                curr_pos = new_curr_pos

                (
                    new_curr_neg,
                    tracking_neg[labels[count - 1] + "_" + subevent],
                ) = self.find_subcategories_neg(
                    curr_neg,
                    cell_ids_activity[subevent]["+_cells"],
                    cell_ids_activity[subevent]["-_cells"],
                    cell_ids_activity[subevent]["N_cells"],
                )

                curr_neg = new_curr_neg

                (
                    new_curr_n,
                    tracking_n[labels[count - 1] + "_" + subevent],
                ) = self.find_subcategories_n(
                    curr_n,
                    cell_ids_activity[subevent]["+_cells"],
                    cell_ids_activity[subevent]["-_cells"],
                    cell_ids_activity[subevent]["N_cells"],
                )

                curr_n = new_curr_n

        # print("CURR RESP:", len(curr_resp))
        print("Labels:", labels)
        print("+ count:", pos_count)
        print("- count:", neg_count)
        print("N count:", n_count)
        print("TRACKING +")
        print(tracking_pos)
        print("TRACKING -")
        print(tracking_neg)
        print("TRACKING NEUTRAL")
        print(tracking_n)


def stacked_barplot_chain(list_1, list_2, labels, dst):
    """labels_input = []

    print("Enter label (presss 'q' to exit):")
    ans = None

    while ans != "q":
        ans = str(input())
        if ans != "q":
            labels_input.append(ans)
        else:
            continue"""

    # Now have labels list (based on my input)

    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(labels, list_1, width, label="Responsive")
    ax.bar(labels, list_2, width, bottom=list_1, label="Non-Responsive")

    ax.set_ylabel("# Cells")
    ax.set_title(" ".join(labels))
    ax.legend()

    plt.savefig(dst)


def stacked_barplot_overall(list_1, list_2, labels, dst):

    # Now have labels list (based on my input)

    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(labels, list_1, width, label="Responsive")
    ax.bar(labels, list_2, width, bottom=list_1, label="Non-Responsive")

    ax.set_ylabel("# Cells")
    ax.set_title(" ".join(labels))
    ax.legend()

    plt.savefig(dst)


def main():
    # select dir where db's exist
    os.chdir("/home/rory/Rodrigo/Database")
    # select where plots will go
    dst = r"/media/rory/Padlock_DT/BLA_Analysis/VennDiagrams"
    # Following parameters determines how many results we acquire from the same data
    dbs = ["BLA_Cells_Ranksum_Pre_Activity", "BLA_Cells_Ranksum_Post_Activity"]
    session_types = ["RDT_D1", "RDT_D2"]
    analysis = "mannwhitneyu"

    # this is where you customize when chain of events to partake in
    # selecting specific columns to start off with, then each subsequent one wil be pulling
    # Block_Reward_Size_Shock_Ocurred_Choice_Time

    ###### Individually give name of columns you want for this analysis ######
    # example column: mannwhitneyu_Shock_Ocurred_Choice_Time_s_True_106_minus10_to_minus5_minus3_to_0

    param_to_compare_subevents = {"Block": ["1dot0", "2dot0", "3dot0"]}

    for param in param_to_compare_subevents["Block"]:

        subevent_chain = [
            "Shock_Ocurred_Choice_Time_s_True",
            "Reward_Size_Choice_Time_s_Large",
            f"Block_Choice_Time_s_{param}",
        ]

        chain_subevent_labels = [
            "Shock_Ocurred_Choice_Time_s_True",
            "Shock_Ocurred_Choice_Time_s_True_Reward_Size_Choice_Time_s_Large",
            f"Shock_Ocurred_Choice_Time_s_True_Reward_Size_Choice_Time_s_Large_Block_Choice_Time_s_{param}",
        ]

        overall_plot_name = f"overall_{'_'.join(subevent_chain)}.png"
        chain_plot_name = f"chain_{'_'.join(subevent_chain)}.png"

        for db in dbs:
            os.chdir("/home/rory/Rodrigo/Database")
            print(f"Curr db: {db}")
            conn = sqlite3.connect(f"{db}.db")
            c = conn.cursor()
            for session in session_types:
                print(f"Curr session: {session}")
                sql_query = pd.read_sql_query(f"SELECT * FROM {session}", conn)
                session_df = pd.DataFrame(sql_query)

                new_subdir = f"{db}/{session}/{'/'.join(subevent_chain)}"
                new_dir = os.path.join(dst, new_subdir)
                os.makedirs(new_dir, exist_ok=True)

                os.chdir(new_dir)

                obj = StreamProportions(
                    session_df, c, db, session, analysis, "Choice_Time", subevent_chain,
                )

                (
                    resp_count,
                    nonresp_count,
                    resp_chain,
                    nonresp_chain,
                ) = obj.stream_responsiveness()

                print(resp_count)
                print(nonresp_count)
                print(resp_chain)
                print(nonresp_chain)

                stacked_barplot_overall(
                    resp_count,
                    nonresp_count,
                    subevent_chain,
                    f"{new_dir}/{overall_plot_name}",
                )

                stacked_barplot_chain(
                    resp_chain,
                    nonresp_chain,
                    chain_subevent_labels,
                    f"{new_dir}/{chain_plot_name}",
                )

            conn.close()


if __name__ == "__main__":
    main()
