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

def pull_subevent_from_string(mystring):
    #Block_Reward_Size_Choice_Time_s_1dot0_Large
    #"(1.0, 'Large')"
    d = {
        "1dot0_Large_False" : "(1.0, 'Large', False)",
        "2dot0_Large_False" : "(2.0, 'Large', False)",
        "3dot0_Large_False" : "(3.0, 'Large', False)",
        "1dot0_Small_False" : "(1.0, 'Small', False)",
        "2dot0_Small_False" : "(2.0, 'Small', False)",
        "3dot0_Small_False" : "(3.0, 'Small', False)",
    }
    mystring = "_".join(mystring.split("_")[8:len(mystring.split("_")) + 1])
    
    return d[mystring]

def pull_block_from_string(mystring):
    #Block_Reward_Size_Choice_Time_s_1dot0_Large
    #"(1.0, 'Large')"
    d = {
        "1dot0_Large_False" : "1.0",
        "2dot0_Large_False" : "2.0",
        "3dot0_Large_False" : "3.0",
        "1dot0_Small_False" : "1.0",
        "2dot0_Small_False" : "2.0",
        "3dot0_Small_False" : "3.0",
    }
    mystring = "_".join(mystring.split("_")[8:len(mystring.split("_")) + 1])
    
    return d[mystring]

def main():

    # BASED ON SHOCK TRUE, REGARDLESS OF BLOCK
    shock_excited_cells = ["9_C09", "1_C11", "6_C06", "14_C07", "6_C19"]
    shock_inhibited_cells = ["1_C08", "13_C06", "13_C02", "6_C07", "9_C06"]
    shock_neutral_cells = ["15_C05", "1_C04", "1_C12", "15_C23", "6_C10"]
    all_cells = shock_excited_cells + shock_inhibited_cells + shock_neutral_cells
    dst = r"/media/rory/Padlock_DT/Rodrigo/Database/VennDiagrams_StackedPlots/results_2"
    os.makedirs(dst, exist_ok=True)

    # Will have to connect to two dbs: post for shock responsive and pre for L/S reward responsive
    specifics = "NOBONF_NOAUC_-10_2"
    db_post = f"/media/rory/Padlock_DT/BLA_Analysis/Database/BLA_Cells_Post_Activity_{specifics}.db"

    conn = sqlite3.connect(db_post)

    sessions = ["RDT_D1",]
    for session in sessions:
        print(F"CURRENT SESSION: {session}")
        session_mod = session.replace("_"," ")

        concat_cells_file_shock = f"/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/{session_mod}/Shock Ocurred_Choice Time (s)/True/all_concat_cells_z_fullwindow.csv"
        event = "Block_Reward_Size_Shock_Ocurred_Choice_Time_s"
        blocks = ["1dot0", "2dot0", "3dot0"]
        outcome = ["Large", "Small"]

        col_names_formatted = ""
        for b in blocks:
            for o in outcome:
                col_name = "_".join([event, b, o, "False"])
                if b == blocks[-1] and o == outcome[-1]:
                    col_names_formatted += col_name
                else:
                    col_names_formatted += col_name + ", "

        chosen_cells_formatted = "("
        for c in all_cells:
            if c == all_cells[-1]:
                chosen_cells_formatted += f"'{c}')"
            else:
                chosen_cells_formatted += f"'{c}', "
        
        query = f"SELECT cell_name, {col_names_formatted} FROM {session} WHERE cell_name IN {chosen_cells_formatted}"

        sql_query = pd.read_sql_query(query, conn)
        df = pd.DataFrame(sql_query)
        df.index = list(df["cell_name"])
        df = df.iloc[:, 1:]

        conn.close()

        df_d = df.to_dict()
        
        final_cell_classifications = {}
        subevent_count = 0

        fig = plt.figure(figsize=(15, 8))
        outer = gridspec.GridSpec(1, 3, wspace=0.0, hspace=0.0)
        t_pos = np.arange(0.0, 10.0, 0.1)
        t_neg = np.arange(-10.0, 0.0, 0.1)
        t = t_neg.tolist() + t_pos.tolist()
        t = [round(i, 1) for i in t]

        fig_2 = plt.figure()
        outer_2 = gridspec.GridSpec(1, 1, wspace=0.0, hspace=0.0)

        for count, subevent in enumerate(df_d):
            inner_idx = None
            if "Large" in subevent: # only want to plot this rew type for now
                curr_mod_subevent = pull_subevent_from_string(subevent)
                print(curr_mod_subevent)
                block = pull_block_from_string(subevent)

                for cell in df_d[subevent]:
                    if subevent in final_cell_classifications:
                        final_cell_classifications[subevent][cell] = df_d[subevent][cell]
                    else:
                        # if the block doesnt exist
                        final_cell_classifications[subevent] = {}
                        final_cell_classifications[subevent][cell] = df_d[subevent][cell]
                
                groups = {
                "Excited" : list(cell for cell in list(final_cell_classifications[subevent].keys()) if final_cell_classifications[subevent][cell] == "+"),
                "Inhibited" : list(cell for cell in list(final_cell_classifications[subevent].keys()) if final_cell_classifications[subevent][cell] == "-"),
                "Neutral" : list(cell for cell in list(final_cell_classifications[subevent].keys()) if final_cell_classifications[subevent][cell] == "Neutral"),
                }

                concat_cells_root = f"/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData/{session_mod}"
                
                concat_cells_file = find_file(concat_cells_root, f"{curr_mod_subevent}/all_concat_cells_z_fullwindow.csv")[0]

                df = pd.read_csv(concat_cells_file)
                df.columns = [col.replace("BLA-Insc-", "") for col in list(df.columns)]
                df = df[all_cells] # color code based on grouping

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
                ax.set_title(curr_mod_subevent)
                ax.set_yticks([])
                if subevent_count == 1:
                    ax.set_ylabel("Neuron #")
                if subevent_count == 2:
                    ax.set_xlabel("Time relative to reward choice (s)")
                ax.get_xaxis().set_visible(True) # doesn't override the off selection of axis

                #print(list(df.columns))
                for col in list(df.columns):
                    colour = None
                    if col in groups["Excited"]:
                        colour = 'red'
                    elif col in groups["Inhibited"]:
                        colour = 'blue'
                    else:
                        colour = 'green'

                    ax.plot(t, list(df[col]), marker='', color=colour, linewidth=2, label=col)
                    if subevent_count == 1:
                        ax.text(-10.0, list(df[col])[0] + 2, col)
                fig.add_subplot(ax)

        df_shock = pd.read_csv(concat_cells_file_shock)
        df_shock.columns = [col.replace("BLA-Insc-", "") for col in list(df_shock.columns)]
        df_shock = df_shock[all_cells] # color code

        for count, col in enumerate(list(df_shock.columns)):
                if count > 0:
                    for idx in range(len(list(df_shock.index))):
                        df_shock.at[idx, col] = df_shock.at[idx, col] + count*4

        inner_2 = gridspec.GridSpecFromSubplotSpec(1, 1,
            subplot_spec=outer_2[0], wspace=0.0, hspace=0.0)
        ax = plt.Subplot(fig_2, inner_2[0])
        #fig.patch.set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        #ax.set_title("Excited: " + subevent)
        ax.set_yticks([])
        #ax.set_ylabel("Neuron #")
        ax.get_xaxis().set_visible(True) # doesn't override the off selection of axis

        #print(df.head())
        for col in list(df_shock.columns):
            colour = None
            if col in shock_excited_cells:
                colour = 'red'
            elif col in shock_inhibited_cells:
                colour = 'blue'
            else:
                colour = 'green'
            
            ax.plot(t, list(df_shock[col]), marker='', color=colour, linewidth=2, label=col)
            ax.text(-10.0, list(df[col])[5] + .5, col)
        fig_2.add_subplot(ax)


        #plt.legend()
        
        dst = f"/media/rory/Padlock_DT/BLA_Analysis/Results/shock_true_example_cells_tracking_large_shockfalse_{specifics}.png"
        fig.savefig(dst)
        dst_2 = f"/media/rory/Padlock_DT/BLA_Analysis/Results/shock_true_example_cells_tracking_{specifics}.png"
        fig_2.savefig(dst_2)

        #plot shock outcomes for these same cells, across blocks, you need to finds it's concat cells file based on what block it is

        break #just doing 1 rdt day

if __name__ == "__main__":
    main()