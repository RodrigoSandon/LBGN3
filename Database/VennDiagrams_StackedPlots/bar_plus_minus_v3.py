from genericpath import exists
from re import sub
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import os

def main():

    dst = r"/media/rory/Padlock_DT/BLA_Analysis/Results/VennDiagrams/results_3"
    os.makedirs(dst, exist_ok=True)

    # Will have to connect to two dbs: post for shock responsive and pre for L/S reward responsive

    db_pre = "/media/rory/Padlock_DT/BLA_Analysis/Database/BLA_Cells_Pre_Activity_BONF_AUC_-8_-5_-3_0.db"

    sessions = ["RDT_D1", ]
    for session in sessions:

        print(F"CURRENT SESSION: {session}")
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

        subset_of_cells_query = f"SELECT cell_name, {col_names_formatted} FROM {session}"
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

        ax.set_ylim(0, 75)

        ax.set_title(f"{session}: Pre-Choice Cells Identities")
        plt.savefig(os.path.join(dst, f"bar_{session}_prechoice_rew_id.png"))
        plt.close()

if __name__ == "__main__":
    main()