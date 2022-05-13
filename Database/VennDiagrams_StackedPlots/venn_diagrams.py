from genericpath import exists
from re import sub
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import os

from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
from matplotlib_venn import venn3_circles
import circlify

def main():

    dst = r"/media/rory/Padlock_DT/Rodrigo/Database/VennDiagrams_StackedPlots/results_3"
    os.makedirs(dst, exist_ok=True)

    # Will have to connect to two dbs: post for shock responsive and pre for L/S reward responsive
    db_post = "/media/rory/Padlock_DT/BLA_Analysis/Database/BLA_Cells_Post_Activity.db"
    db_pre = "/media/rory/Padlock_DT/BLA_Analysis/Database/BLA_Cells_Pre_Activity.db"

    sessions = ["RDT_D1", "RDT_D2", "RDT_D3"]
    for session in sessions:

        print(F"CURRENT SESSION: {session}")

        conn = sqlite3.connect(db_post)
        cursor = conn.cursor()

        shock_col_name = "Shock_Ocurred_Choice_Time_s_True"
        query = f"SELECT cell_name, {shock_col_name} FROM {session} WHERE {shock_col_name} = '+' OR {shock_col_name} = '-'"
        query_2 = f"SELECT cell_name, {shock_col_name} FROM {session} WHERE {shock_col_name} = 'Neutral'"

        sql_query = pd.read_sql_query(query, conn)
        sql_query2 = pd.read_sql_query(query_2, conn)
        df_1 = pd.DataFrame(sql_query)
        df_2 = pd.DataFrame(sql_query2)

        chosen_cells = list(df_1["cell_name"])
        num_chosen_cells = len(chosen_cells)
        neu_cells = list(df_2["cell_name"])
        num_neu_cells = len(neu_cells)
        
        chosen_cells_formatted = "("

        ##### CHANGE BETWEEN TYPES OF CHOSEN CELLS (RESP OR NONRESP) TO ANALYZE #####
        for c in neu_cells:
            if c == neu_cells[-1]: #at last one
                chosen_cells_formatted += f"'{c}')"
            else:
                chosen_cells_formatted += f"'{c}', "

        #print(chosen_cells_formatted)
        conn.close()

        # Get pie chart for shock activated cells

        fig = plt.figure(figsize=(10, 7))
        plt.pie([num_chosen_cells, num_neu_cells], labels=[f"Shock Responsive-{num_chosen_cells}", f"Non-Responsive-{num_neu_cells}"], autopct="%1.2f%%")
        plt.title(f"{session}: Shock Responsiveness")
        plt.savefig(os.path.join(dst, f"shock_resp_{session}.png"))
        plt.close()

        conn = sqlite3.connect(db_pre)
        cursor = conn.cursor()

        event = "Block_Reward_Size_Choice_Time_s"
        blocks = ["1dot0", "2dot0", "3dot0"]
        outcome = ["Large", "Small"]

        col_names_formatted = ""
        for b in blocks:
            for o in outcome:
                col_name = "_".join([event, b, o])
                if b == blocks[-1] and o == outcome[-1]: # at last one
                    col_names_formatted += col_name
                else:
                    col_names_formatted += col_name + ", "
        # have col names of all subevents we want, now get them
        
        #print(col_names_formatted)
        query = f"SELECT cell_name, {col_names_formatted} FROM {session} WHERE cell_name IN {chosen_cells_formatted}"

        sql_query = pd.read_sql_query(query, conn)
        df_2 = pd.DataFrame(sql_query)
        df_2.index = list(df_2["cell_name"])
        df_2 = df_2.iloc[:, 1:]
        #print(df_2)
        #print(list(df_2.columns))
        conn.close()

        df_2_d = df_2.to_dict()
        #print(df_2_d)
        # Now we have prechoice cell IDs for only the cells that we're shock responsive in post activity db
        # block: cell_id : only L responsive/ only S responsive/ Both
        final_cell_classifications = {}
        for block in blocks:
            subevent_count = 0

            for subevent in df_2_d:
                
                if block in subevent:
                    # now we have 1 subevent id, first just store it
                    # then on the second subevent seen we determine
                    # the cell's final classification
                    print(subevent)
                    subevent_count += 1
                    if subevent_count == 1:
                        # just store og cell id's
                        for cell in df_2_d[subevent]:
                            if block in final_cell_classifications:
                                final_cell_classifications[block][cell] = df_2_d[subevent][cell]
                            else:
                                # if the block doesnt exist
                                final_cell_classifications[block] = {}
                                final_cell_classifications[block][cell] = df_2_d[subevent][cell]

                    elif subevent_count > 1: # actually give the cell a new classification
                        # all keys exist
                        for cell in df_2_d[subevent]:
                            prev_id_L = final_cell_classifications[block][cell]
                            
                            curr_id_S = df_2_d[subevent][cell]

                            # if one not neutral but one is, only rresponsive to L/S
                            # if both not neutral, then cel responsive to both
                            if prev_id_L != "Neutral" and curr_id_S == "Neutral":
                                final_cell_classifications[block][cell] = "Large Responsive"
                            elif prev_id_L == "Neutral" and curr_id_S != "Neutral":
                                final_cell_classifications[block][cell] = "Small Responsive"
                            elif prev_id_L != "Neutral" and curr_id_S != "Neutral":
                                final_cell_classifications[block][cell] = "Dual Responsive"
                            elif prev_id_L == "Neutral" and curr_id_S == "Neutral":
                                final_cell_classifications[block][cell] = "Non-Responsive"

        """for key, value in final_cell_classifications.items():
            print(f"{key} : {value}")"""

        final_cell_classifications_arranged = {

        }


        for key, value in final_cell_classifications.items():
            final_cell_classifications_arranged[key] = {
            "Large Responsive": list(),
            "Small Responsive": list(),
            "Dual Responsive": list(),
            "Non-Responsive": list()
        }
        
        #all keys exist, ready to append cells
        for key in final_cell_classifications:
            for cell, id in final_cell_classifications[key].items():
                final_cell_classifications_arranged[key][id].append(cell)

        for key, value in final_cell_classifications_arranged.items():
            print(f"{key} : {value}")

        for key in final_cell_classifications_arranged:
            num_l = len(final_cell_classifications_arranged[key]["Large Responsive"])
            num_s = len(final_cell_classifications_arranged[key]["Small Responsive"])
            num_dual = len(final_cell_classifications_arranged[key]["Dual Responsive"])
            num_non = len(final_cell_classifications_arranged[key]["Non-Responsive"])

            plt.figure()
            print(key)
            if num_s != 0:
                venn3(
                    subsets=(num_l, num_s, num_dual, num_non, 0, 0, 0),
                    set_labels=("Large", "Small", "Non-Responsive"),
                )
            else:
                venn2(
                    subsets=(num_l, num_non, 0),
                    set_labels=("Large", "Non-Responsive"),
                )

            plt.title(f"{session}: Identity Proportions of Shock Non-Responsive Cells in {key}")
            plt.savefig(os.path.join(dst, f"venn_shock_nonresp_{session}_{key}.png"))
            plt.close()

if __name__ == "__main__":
    main()
