"""
This will create final figures for the prezi:

1. Stacked bar plot of cell classifications for each block

"""

from genericpath import exists
from re import sub
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import os, re

from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
from matplotlib_venn import venn3_circles

import numpy as np
from scipy.stats import chi2_contingency

def perform_chi_square_test(session_dicts):
    keys = ['+', '-', 'Neutral']
    observed = np.array([[len(d[key]) for key in keys] for d in session_dicts])
    chi2, p, _, _ = chi2_contingency(observed)
    return p

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

def search_columns(df, column_names):
    for col in df.columns:
        if all(elem in col for elem in column_names):
            return col
    return None

def main():

    dst = r"/media/rory/Padlock_DT/Rodrigo/Database/VennDiagrams_StackedPlots/results_2"
    os.makedirs(dst, exist_ok=True)

    # Will have to connect to two dbs: post for shock responsive and pre for L/S reward responsive
    db_post = "/media/rory/Padlock_DT/Rodrigo/Database/BLA_Cells_Ranksum_Post_Activity.db"
    timewindow_post = "minus10_to_minus5_0_to_3"

    db_pre = "/media/rory/Padlock_DT/Rodrigo/Database/BLA_Cells_Ranksum_Consumption_Identities.db"
    timewindow_pre = "minus10_to_minus5_1_to_4"

    """db_pre = "/media/rory/Padlock_DT/Rodrigo/Database/BLA_Cells_Ranksum_Pre_Activity.db"
    timewindow_pre = "minus10_to_minus5_minus3_to_0"""

    comparison_test = "mannwhitneyu"
    sessions = ["Pre_RDT_RM","RDT_D1", "RDT_D2", "RDT_D3"]
    shock_resp = False
    #sessions = ["RDT_D1"]
    # you're gonna have just one figure
    # Inside main function
    fig, axs = plt.subplots(1, len(sessions), figsize=(15, 5))
    fig.tight_layout(pad=7)

    for i, session in enumerate(sessions):
        print(F"CURRENT SESSION: {session}")

        conn = sqlite3.connect(db_post)
        cursor = conn.cursor()

        # Getting number of rows
        df = pd.read_sql_query(f"SELECT * FROM {session}", conn)
        len_df = len(df)
        #print(len(df))
        shock_happened = "False"
        shock_col_name = f"{comparison_test}_Shock_Ocurred_Choice_Time_s_{shock_happened}_{len_df}_{timewindow_post}"
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
        # neu cells - nonresp
        # chosen cells - resp
        if shock_resp == True:
            cells = chosen_cells
        else:
            cells = neu_cells

        for c in cells:
            if c == cells[-1]: #at last one
                chosen_cells_formatted += f"'{c}')"
            else:
                chosen_cells_formatted += f"'{c}', "

        #print(chosen_cells_formatted)
        conn.close()

        conn = sqlite3.connect(db_pre)
        df = pd.read_sql_query(f"SELECT * FROM {session}", conn)
        len_df = len(df)
        cursor = conn.cursor()

        """event = "Block_Reward_Size_Choice_Time_s"
        blocks = ["1dot0", "2dot0", "3dot0"]
        outcome = ["Large", "Small"]"""
        event = "Block_Choice_Time_s"
        blocks = ["1dot0", "2dot0", "3dot0"]

        col_names_formatted = ""
        for b in blocks:

            col_name = "_".join([comparison_test, event, b,f"{len_df}", timewindow_pre])
            # find ideal col
            actual_col = search_columns(df, [comparison_test, event, b, timewindow_pre])
            print(actual_col)
            if b == blocks[-1]: # at last one
                col_names_formatted += actual_col
            else:
                col_names_formatted += actual_col + ", "
        

        query = f"SELECT cell_name, {col_names_formatted} FROM {session} WHERE cell_name IN {chosen_cells_formatted}"

        sql_query = pd.read_sql_query(query, conn)
        df_3 = pd.DataFrame(sql_query)
        df_3.index = list(df_3["cell_name"])
        df_3 = df_3.iloc[:, 1:]
        #print(df_2)
        #print(list(df_2.columns))
        conn.close()

        df_3_d = df_3.to_dict()
        print("df_3_d")
        print(df_3_d)


        """for key, value in final_cell_classifications.items():
            print(f"{key} : {value}")"""

        final_cell_classifications_arranged = {

        }

        for key, value in df_3_d.items():
            final_cell_classifications_arranged[key] = {
            "+": list(),
            "-": list(),
            "Neutral": list(),
        }
        

        #all keys exist, ready to append cells
        for key in df_3_d:
            for cell, id in df_3_d[key].items():
                final_cell_classifications_arranged[key][id].append(cell)

        print("final_cell_classifications_arranged")
        for key, value in final_cell_classifications_arranged.items():
            print(f"{key} : {value}")

        width = 0.35
        ax = axs[i]  # Change this line

        # make stacked bar plot
        # each block's category needs to be zipped bc the bars will all be
        # layed out at once
        # therefore need to zip it

        # each responsiveness category will be zipped for 3 blocks 
        # #(3 elements in each responsivenes category)
        zips = {
            "+": list(),
            "-": list(),
            "Neutral": list(),
        }

        x_labels = ["1", "2", "3"]

        for key in final_cell_classifications_arranged:
            num_l = len(final_cell_classifications_arranged[key]["+"])
            zips["+"].append(num_l)
            num_s = len(final_cell_classifications_arranged[key]["-"])
            zips["-"].append(num_s)
            num_dual = len(final_cell_classifications_arranged[key]["Neutral"])
            zips["Neutral"].append(num_dual)

        zips_2 = zip(zips["+"], zips["-"], zips["Neutral"])
        zips_3 = zip(zips["-"], zips["Neutral"])

        # will be top
        plus = [x + y for (x, y) in zips_3]
        neg = zips["Neutral"]
        neutral = 0

        ax.bar(x_labels, zips["Neutral"], width, bottom=neutral, label="Neutral", color="dimgrey")
        ax.bar(x_labels, zips["-"], width, bottom=neg, label="-", color="royalblue")
        ax.bar(x_labels, zips["+"], width, bottom=plus, label="+", color="indianred")

    ax.set_ylabel("# Cells")
    ax.set_xlabel("Block")
    #ax.set_title(f"Identity Proportions of Shock Non-Responsive Cells Across Blocks, Across Sessions")
    ax.legend(loc='center', bbox_to_anchor=(0.5, -0.2,),
        fancybox=True, shadow=True, ncol=4, borderpad=0.3, fontsize="small")
    
    if shock_resp == True:
        label = "respshock"
    else:
        label = "nonrespshock"

    plot_path = os.path.join(dst, f"stacked_plot_{label}_{session}_{timewindow_post}_{timewindow_pre}_all.png")
    plt.savefig(plot_path)



if __name__ == "__main__":
    main()
