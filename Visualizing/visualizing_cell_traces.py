from genericpath import exists
from re import sub
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import List
import os, glob
import numpy as np
import random


from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
from matplotlib_venn import venn3_circles
import matplotlib.gridspec as gridspec

def find_file(root_path: Path, endswith: str) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, "**", endswith), recursive=True,
    )
    return files

def venn_diagram(list_1, list_2, list_3, title, labels, dst):

    # Now have labels list (based on my input)

    for i in range(0, 3):
        plt.figure()
        print(i)
        venn2(
            subsets=(list_1[i], list_3[i], list_2[i]),
            set_labels=("Large", "Small", "Both"),
        )

        plt.title(title)
        dst = dst.replace(".csv", f"_block{i+1}.png")
        plt.show()
        plt.savefig(dst)
        plt.close()

# large, small, dual, non
def stacked_barplot(list_1, list_2, list_3, list_4, title, labels, dst):

    # Now have labels list (based on my input)

    width = 0.35
    fig, ax = plt.subplots()

    zipped = zip(list_4, list_3, list_2)

    sum = [x + y for (x, y) in zipped]

    ax.bar(labels, list_1, width, label="Non-Responsive")
    ax.bar(labels, list_2, width, bottom=list_1, label="S")
    ax.bar(labels, list_3, width, bottom=sum, label="Small Responsive")

    ax.set_ylabel("# Cells")
    ax.set_title(title)
    ax.legend()

    plt.savefig(dst)

def pull_subevent_from_string(mystring):
    #Block_Reward_Size_Choice_Time_s_1dot0_Large
    #"(1.0, 'Large')"
    d = {
        "1dot0_Large" : "(1.0, 'Large')",
        "2dot0_Large" : "(2.0, 'Large')",
        "3dot0_Large" : "(3.0, 'Large')",
        "1dot0_Small" : "(1.0, 'Small')",
        "2dot0_Small" : "(2.0, 'Small')",
        "3dot0_Small" : "(3.0, 'Small')",
    }
    mystring = "_".join(mystring.split("_")[6:len(mystring.split("_")) + 1])
    
    return d[mystring]

def main():

    dst = r"/media/rory/Padlock_DT/Rodrigo/Database/VennDiagrams_StackedPlots/results_2"
    os.makedirs(dst, exist_ok=True)

    # Will have to connect to two dbs: post for shock responsive and pre for L/S reward responsive
    db_post = "/media/rory/Padlock_DT/BLA_Analysis/Database/BLA_Cells_Post_Activity.db"
    db_pre = "/media/rory/Padlock_DT/BLA_Analysis/Database/BLA_Cells_Pre_Activity.db"

    sessions = ["RDT_D1", "RDT_D2", "RDT_D3"]
    for session in sessions:
        print(F"CURRENT SESSION: {session}")
        session_mod = session.replace("_"," ")

        conn = sqlite3.connect(db_post)
        cursor = conn.cursor()

        shock_col_name = "Shock_Ocurred_Choice_Time_s_True"
        query = f"SELECT cell_name, {shock_col_name} FROM {session} WHERE {shock_col_name} = '+' OR {shock_col_name} = '-'"
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
        df_3 = pd.DataFrame(sql_query)
        df_3.index = list(df_3["cell_name"])
        df_3 = df_3.iloc[:, 1:]
        print(df_3.head())
        conn.close()

        df_3_d = df_3.to_dict()
        
        final_cell_classifications = {}
        subevent_count = 0

        fig = plt.figure(figsize=(15, 8))
        outer = gridspec.GridSpec(1, 3, wspace=0.0, hspace=0.0)
        t_pos = np.arange(0.0, 10.0, 0.1)
        t_neg = np.arange(-10.0, 0.0, 0.1)
        t = t_neg.tolist() + t_pos.tolist()
        t = [round(i, 1) for i in t]

        for subevent in df_3_d:
            if "Large" in subevent: # only want to plot this rew type for now 
                #subevent_count += 1
                curr_mod_subevent = pull_subevent_from_string(subevent)
                print(curr_mod_subevent)

                for cell in df_3_d[subevent]:
                    if subevent in final_cell_classifications:
                        final_cell_classifications[subevent][cell] = df_3_d[subevent][cell]
                    else:
                        # if the block doesnt exist
                        final_cell_classifications[subevent] = {}
                        final_cell_classifications[subevent][cell] = df_3_d[subevent][cell]

                #print(final_cell_classifications)

                for key, value in final_cell_classifications[subevent].items():
                    #final_cell_classifications[subevent]["BLA-Insc-" + key] = final_cell_classifications[subevent].pop(key)
                    print(f"{key} : {value}")
                
                groups = {
                "Excited" : list(cell for cell in list(final_cell_classifications[subevent].keys()) if final_cell_classifications[subevent][cell] == "+"),
                "Inhibited" : list(cell for cell in list(final_cell_classifications[subevent].keys()) if final_cell_classifications[subevent][cell] == "-"),
                "Neutral" : list(cell for cell in list(final_cell_classifications[subevent].keys()) if final_cell_classifications[subevent][cell] == "Neutral"),
                }

                concat_cells_root = f"/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/{session_mod}"

                concat_cells_file = find_file(concat_cells_root, f"{curr_mod_subevent}/all_concat_cells_z_fullwindow.csv")[0]
                print(concat_cells_file)

                df = pd.read_csv(concat_cells_file)
                df.columns = [col.replace("BLA-Insc-", "") for col in list(df.columns)]
                
                #Choose which type of cells, only get X amount of cells (random)
                df = df[groups["Excited"]]

                rand_cells = list(random.sample(list(df.columns), k=4))
                
                df = df[rand_cells]

                for count, col in enumerate(list(df.columns)):
                    if count > 0:
                        for idx in range(len(list(df.index))):
                            df.at[idx, col] = df.at[idx, col] + count*4

        
        
                inner = gridspec.GridSpecFromSubplotSpec(1, 1,
                    subplot_spec=outer[subevent_count], wspace=0.0, hspace=0.0)
                subevent_count += 1

                ax = plt.Subplot(fig, inner[0])
                #fig.patch.set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.set_title("Excited: " + subevent)
                ax.set_yticks([])
                ax.set_ylabel("Neuron #")
                ax.get_xaxis().set_visible(True) # doesn't override the off selection of axis

                print(df.head())
                for col in list(df.columns):
                    print(t)
                    print(list(df[col]))
                    ax.plot(t, list(df[col]), marker='', color='black', linewidth=2)
                fig.add_subplot(ax)

        #plt.legend()
        dst = f"/media/rory/Padlock_DT/BLA_Analysis/Results/ex.png"
        plt.savefig(dst)
        break #just doing 1 rdt day

if __name__ == "__main__":
    main()