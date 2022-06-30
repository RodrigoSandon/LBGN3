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

    dst = r"/media/rory/Padlock_DT/BLA_Analysis/Results/VennDiagrams/results_3"
    os.makedirs(dst, exist_ok=True)

    # Will have to connect to two dbs: post for shock Inhibited and pre for L/S reward Inhibited
    db_post = "/media/rory/Padlock_DT/BLA_Analysis/Database/BLA_Cells_Post_Activity_BONF_AUC_-3_0_0_3.db"
    db_pre = "/media/rory/Padlock_DT/BLA_Analysis/Database/BLA_Cells_Pre_Activity_BONF_AUC_-8_-5_-3_0.db"

    sessions = ["RDT_D1", ]
    for session in sessions:

        print(F"CURRENT SESSION: {session}")

        ### GETTING POST-CHOICE SHOCK CELL IDENTITIES
        conn = sqlite3.connect(db_post)
        cursor = conn.cursor()

        shock_col_name = "Shock_Ocurred_Choice_Time_s_True"

        shock_excited = f"SELECT cell_name, {shock_col_name} FROM {session} WHERE {shock_col_name} = '+'"
        shock_inhibited = f"SELECT cell_name, {shock_col_name} FROM {session} WHERE {shock_col_name} = '-'"
        shock_neutral = f"SELECT cell_name, {shock_col_name} FROM {session} WHERE {shock_col_name} = 'Neutral'"

        excited_query = pd.read_sql_query(shock_excited, conn)
        inhibited_query = pd.read_sql_query(shock_inhibited, conn)
        neutral_query = pd.read_sql_query(shock_neutral, conn)

        excited_df = pd.DataFrame(excited_query)
        inhibited_df = pd.DataFrame(inhibited_query)
        neutral_df = pd.DataFrame(neutral_query)

        excited_cells = list(excited_df["cell_name"])
        num_excited_cells = len(excited_cells)

        inhibited_cells = list(inhibited_df["cell_name"])
        num_inhibited_cells = len(inhibited_cells)

        neutral_cells = list(neutral_df["cell_name"])
        num_neutral_cells = len(neutral_cells)
        
        ##### CHANGE BETWEEN TYPES OF CHOSEN CELLS (EXCITED/INHIBITED/NEUTRAL) TO ANALYZE #####
        chosen_cells = neutral_cells
        chosen_cells_name = "Neutral"
        ##### CHANGE BETWEEN TYPES OF CHOSEN CELLS (EXCITED/INHIBITED/NEUTRAL) TO ANALYZE #####

        chosen_cells_formatted = "("
        
        for c in chosen_cells:
            if c == chosen_cells[-1]: #at last one
                chosen_cells_formatted += f"'{c}')"
            else:
                chosen_cells_formatted += f"'{c}', "

        conn.close()

        # Get pie chart for shock activated cells
        fig = plt.figure(figsize=(10, 7))
        plt.pie([num_excited_cells, num_inhibited_cells, num_neutral_cells], labels=[f"Shock Excited-{num_excited_cells}", f"Shock Inhibited-{num_inhibited_cells}", f"Shock Neutral-{num_neutral_cells}"], autopct="%1.2f%%")
        plt.title(f"{session}: Shock Identities")
        plt.savefig(os.path.join(dst, f"shock_identities_{session}.png"))
        plt.close()


        ### ANALYZING POST-CHOICE SHOCK REACTIVITY TO PRE-CHOICE ACTIVITY
        conn = sqlite3.connect(db_pre)
        cursor = conn.cursor()

        event = "Reward_Size_Choice_Time_s"
        outcomes = ["Large", "Small"]

        col_names_formatted = ""
        
        for o in outcomes:
            col_name = "_".join([event, o])
            if o == outcomes[-1]:
                col_names_formatted += col_name
            else:
                col_names_formatted += col_name + ", "

        subset_of_cells_query = f"SELECT cell_name, {col_names_formatted} FROM {session} WHERE cell_name IN {chosen_cells_formatted}"
        # these chosen cells can be +/-/N

        subset_query = pd.read_sql_query(subset_of_cells_query, conn)
        subset_cells_df = pd.DataFrame(subset_query)
        subset_cells_df.index = list(subset_cells_df["cell_name"])
        subset_cells_df = subset_cells_df.iloc[:, 1:]
        
        conn.close()

        subset_cells_d = subset_cells_df.to_dict()
        
        final_cell_classifications = {}

        subevent_count = 0
        for subevent in subset_cells_d:
            
            print(subevent)
            print(subevent_count)
            if subevent_count == 0:
                for cell in subset_cells_d[subevent]:
                    final_cell_classifications[cell] = subset_cells_d[subevent][cell]
                
            elif subevent_count >= 1: # actually give the cell a new classification
                # chosen cells will either be +/-/N identity to shock, this just
                # gives a deeper look into how these identities play out in rew responsiveness
                
                for cell in subset_cells_d[subevent]:
                    prev_id_L = final_cell_classifications[cell]
                    
                    curr_id_S = subset_cells_d[subevent][cell]

                    if prev_id_L == "+" and curr_id_S == "Neutral":
                        final_cell_classifications[cell] = "Large Excited | Small Neutral"

                    elif prev_id_L == "+" and curr_id_S == "-":
                        final_cell_classifications[cell] = "Large Excited | Small Inhibited"

                    elif prev_id_L == "+" and curr_id_S == "+":
                        final_cell_classifications[cell] = "Large and Small Excited"
                    
                    elif prev_id_L == "-" and curr_id_S == "Neutral":
                        final_cell_classifications[cell] = "Large Inhibited | Small Neutral"

                    elif prev_id_L == "-" and curr_id_S == "-":
                        final_cell_classifications[cell] = "Large and Small Inhibited"

                    elif prev_id_L == "-" and curr_id_S == "+":
                        final_cell_classifications[cell] = "Large Inhibited | Small Excited"

                    elif prev_id_L == "Neutral" and curr_id_S == "Neutral":
                        final_cell_classifications[cell] = "Large and Small Neutral"

                    elif prev_id_L == "Neutral" and curr_id_S == "-":
                        final_cell_classifications[cell] = "Large Neutral | Small Inhibited"

                    elif prev_id_L == "Neutral" and curr_id_S == "+":
                        final_cell_classifications[cell] = "Large Neutral | Small Excited"

            subevent_count += 1            

        """for key, value in final_cell_classifications.items():
            print(f"{key} : {value}")"""

        final_cell_classifications_arranged = {
            "Large Excited | Small Neutral": list(),
            "Large Excited | Small Inhibited": list(),
            "Large and Small Excited": list(),

            "Large Inhibited | Small Neutral": list(),
            "Large and Small Inhibited": list(),
            "Large Inhibited | Small Excited": list(),
            
            "Large and Small Neutral": list(),
            "Large Neutral | Small Inhibited": list(),
            "Large Neutral | Small Excited": list()
        }

        #all keys exist, ready to append cells
        for key, value in final_cell_classifications.items():
            final_cell_classifications_arranged[value].append(key)

        for key, value in final_cell_classifications_arranged.items():
            print(f"{key} : {value}")

        
        num_l_ex_s_neu = len(final_cell_classifications_arranged["Large Excited | Small Neutral"])
        num_l_ex_s_in = len(final_cell_classifications_arranged["Large Excited | Small Inhibited"])
        num_l_ex_s_ex = len(final_cell_classifications_arranged["Large and Small Excited"])

        num_l_in_s_neu = len(final_cell_classifications_arranged["Large Inhibited | Small Neutral"])
        num_l_in_s_in = len(final_cell_classifications_arranged["Large and Small Inhibited"])
        num_l_in_s_ex = len(final_cell_classifications_arranged["Large Inhibited | Small Excited"])

        num_l_neu_s_neu = len(final_cell_classifications_arranged["Large and Small Neutral"])
        num_l_neu_s_in = len(final_cell_classifications_arranged["Large Neutral | Small Inhibited"])
        num_l_neu_s_ex = len(final_cell_classifications_arranged["Large Neutral | Small Excited"])
        

        fig, ax = plt.subplots()

        """
        "L+|SN", "Large Excited | Small Neutral"
        "L+|S-", "Large Excited | Small Inhibited"
        "L+|S+", "Large and Small Excited"
        "L-|SN", "Large Inhibited | Small Neutral"
        "L-|S-", "Large and Small Inhibited"
        "L-|S+", "Large Inhibited | Small Excited"
        "LN|SN", "Large and Small Neutral"
        "LN|S-", "Large Neutral | Small Inhibited"
        "LN|S+" "Large Neutral | Small Excited"
        """
        

        labels = ["L+|SN", "L+|S-", "L+|S+", "L-|SN", "L-|S-", "L-|S+", "LN|SN", "LN|S-", "LN|S+",]
        numbers = [num_l_ex_s_neu, num_l_ex_s_in, num_l_ex_s_ex, num_l_in_s_neu, num_l_in_s_in, num_l_in_s_ex, num_l_neu_s_neu, num_l_neu_s_in, num_l_neu_s_ex]

        ax.bar(labels, numbers)
        for i, v in enumerate(numbers):
            ax.text(i, v+1, str(v), color='blue', fontweight='bold')

        ax.set_ylim(0, 35)

        ax.set_title(f"{session}: Identities of Shock {chosen_cells_name} Cells")
        plt.savefig(os.path.join(dst, f"bar_{session}_Shock_{chosen_cells_name}.png"))
        plt.close()

if __name__ == "__main__":
    main()