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
    db_post = "/media/rory/Padlock_DT/BLA_Analysis/Database/BLA_Cells_Post_Activity_BONF_AUC_-3_0_0_3.db"
    db_pre = "/media/rory/Padlock_DT/BLA_Analysis/Database/BLA_Cells_Pre_Activity_BONF_AUC_-8_-5_-3_0.db"

    sessions = ["RDT_D1", ]
    for session in sessions:

        print(F"CURRENT SESSION: {session}")

        conn = sqlite3.connect(db_post)
        cursor = conn.cursor()

        shock_col_name = "Shock_Ocurred_Choice_Time_s_True"
        query = f"SELECT cell_name, {shock_col_name} FROM {session} WHERE {shock_col_name} = '-'  OR {shock_col_name} = '+'"
        query_2 = f"SELECT cell_name, {shock_col_name} FROM {session} WHERE {shock_col_name} = 'Neutral'"

        sql_query = pd.read_sql_query(query, conn)
        sql_query2 = pd.read_sql_query(query_2, conn)
        df_1 = pd.DataFrame(sql_query)
        df_2 = pd.DataFrame(sql_query2)

        resp_cells = list(df_1["cell_name"])
        num_resp_cells = len(resp_cells)
        neu_cells = list(df_2["cell_name"])
        num_neu_cells = len(neu_cells)
        
        chosen_cells_formatted = "("

        ##### CHANGE BETWEEN TYPES OF CHOSEN CELLS (RESP OR NONRESP) TO ANALYZE #####
        for c in resp_cells:
            if c == resp_cells[-1]: #at last one
                chosen_cells_formatted += f"'{c}')"
            else:
                chosen_cells_formatted += f"'{c}', "

        #print(chosen_cells_formatted)
        conn.close()

        # Get pie chart for shock activated cells

        fig = plt.figure(figsize=(10, 7))
        plt.pie([num_resp_cells, num_neu_cells], labels=[f"Shock Responsive-{num_resp_cells}", f"Non-Responsive-{num_neu_cells}"], autopct="%1.2f%%")
        plt.title(f"{session}: Shock Responsiveness")
        plt.savefig(os.path.join(dst, f"shock_resp_{session}.png"))
        plt.close()

        conn = sqlite3.connect(db_pre)
        cursor = conn.cursor()

        event = "Reward_Size_Choice_Time_s"
        outcomes = ["Large", "Small"]

        col_names_formatted = ""
        
        for o in outcomes:
            col_name = "_".join([event, o])
            if o == outcomes[-1]: # at last one
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
        # outcome: cell_id : only L responsive/ only S responsive/ Both
        final_cell_classifications = {}

        subevent_count = 0
        for subevent in df_2_d:
            
            print(subevent)
            
            print(subevent_count)
            if subevent_count == 0:
                # just store og cell id's
                for cell in df_2_d[subevent]:
                    
                    
                    final_cell_classifications[cell] = df_2_d[subevent][cell]
                    
                    
                
            elif subevent_count >= 1: # actually give the cell a new classification
                # all keys exist
                for cell in df_2_d[subevent]:
                    prev_id_L = final_cell_classifications[cell]
                    
                    curr_id_S = df_2_d[subevent][cell]

                    # if one not neutral but one is, only rresponsive to L/S
                    # if both not neutral, then cel responsive to both
                    if prev_id_L != "Neutral" and curr_id_S == "Neutral":
                        final_cell_classifications[cell] = "Large Responsive"
                    elif prev_id_L == "Neutral" and curr_id_S != "Neutral":
                        final_cell_classifications[cell] = "Small Responsive"
                    elif prev_id_L != "Neutral" and curr_id_S != "Neutral":
                        final_cell_classifications[cell] = "Dual Responsive"
                    elif prev_id_L == "Neutral" and curr_id_S == "Neutral":
                        final_cell_classifications[cell] = "Non-Responsive"
            subevent_count += 1            

        """for key, value in final_cell_classifications.items():
            print(f"{key} : {value}")"""

        final_cell_classifications_arranged = {
            "Large Responsive": list(),
            "Small Responsive": list(),
            "Dual Responsive": list(),
            "Non-Responsive": list()
        }

        
        #all keys exist, ready to append cells
        for key, value in final_cell_classifications.items():
            final_cell_classifications_arranged[value].append(key)

        for key, value in final_cell_classifications_arranged.items():
            print(f"{key} : {value}")

        
        num_l = len(final_cell_classifications_arranged["Large Responsive"])
        num_s = len(final_cell_classifications_arranged["Small Responsive"])
        num_dual = len(final_cell_classifications_arranged["Dual Responsive"])
        num_non = len(final_cell_classifications_arranged["Non-Responsive"])

        plt.figure()
        #print(key)
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

        plt.title(f"{session}: Identities of Shock NonResponsive Cells")
        plt.savefig(os.path.join(dst, f"venn_{session}_Shock_NonResponsive_v2.png"))
        plt.close()

if __name__ == "__main__":
    main()