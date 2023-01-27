from genericpath import exists
from re import sub
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import os

from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
from matplotlib_venn import venn3_circles

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

def main():

    dst = r"/media/rory/Padlock_DT/Rodrigo/Database/VennDiagrams_StackedPlots/results_2"
    os.makedirs(dst, exist_ok=True)

    # Will have to connect to two dbs: post for shock responsive and pre for L/S reward responsive
    db_post = "/media/rory/Padlock_DT/Rodrigo/Database/BLA_Cells_Ranksum_Post_Activity.db"
    timewindow_post = "minus10_to_minus5_0_to_3"

    db_pre = "/media/rory/Padlock_DT/Rodrigo/Database/BLA_Cells_Ranksum_Pre_Activity.db"
    timewindow_pre = "minus10_to_minus5_minus3_to_0"

    comparison_test = "mannwhitneyu"
    sessions = ["RDT_D1", "RDT_D2", "RDT_D3"]
    shock_resp = False
    #sessions = ["RDT_D1"]
    for session in sessions:
        print(F"CURRENT SESSION: {session}")

        conn = sqlite3.connect(db_post)
        cursor = conn.cursor()

        # Getting number of rows
        df = pd.read_sql_query(f"SELECT * FROM {session}", conn)
        len_df = len(df)
        #print(len(df))

        shock_col_name = f"{comparison_test}_Shock_Ocurred_Choice_Time_s_True_{len_df}_{timewindow_post}"
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
        cursor = conn.cursor()

        event = "Block_Reward_Size_Choice_Time_s"
        blocks = ["1dot0", "2dot0", "3dot0"]
        outcome = ["Large", "Small"]

        col_names_formatted = ""
        for b in blocks:
            for o in outcome:
                col_name = "_".join([comparison_test, event, b, o,f"{len_df}", timewindow_pre])
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
        #print(df_2)
        #print(list(df_2.columns))
        conn.close()

        df_3_d = df_3.to_dict()
        #print(df_2_d)
        # Now we have prechoice cell IDs for only the cells that we're shock responsive in post activity db
        # block: cell_id : only L responsive/ only S responsive/ Both
        final_cell_classifications = {}
        for block in blocks:
            subevent_count = 0

            for subevent in df_3_d:
                
                if block in subevent:
                    # now we have 1 subevent id, first just store it
                    # then on the second subevent seen we determine
                    # the cell's final classification
                    print(subevent)
                    subevent_count += 1
                    if subevent_count == 1:
                        # just store og cell id's
                        for cell in df_3_d[subevent]:
                            if block in final_cell_classifications:
                                final_cell_classifications[block][cell] = df_3_d[subevent][cell]
                            else:
                                # if the block doesnt exist
                                final_cell_classifications[block] = {}
                                final_cell_classifications[block][cell] = df_3_d[subevent][cell]

                    elif subevent_count > 1: # actually give the cell a new classification
                        # all keys exist
                        for cell in df_3_d[subevent]:
                            prev_id_L = final_cell_classifications[block][cell]
                            
                            curr_id_S = df_3_d[subevent][cell]

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

        width = 0.35
        fig, ax = plt.subplots()
        fig.tight_layout(pad=7)

        # make stacked bar plot
        # each block's category needs to be zipped bc the bars will all be
        # layed out at once
        # therefore need to zip it

        # each responsiveness category will be zipped for 3 blocks 
        # #(3 elements in each responsivenes category)
        zips = {
            "Large Responsive" : list(),
            "Small Responsive" : list(),
            "Dual Responsive" : list(),
            "Non-Responsive" : list()
        }

        x_labels = ["1", "2", "3"]

        for key in final_cell_classifications_arranged:
            num_l = len(final_cell_classifications_arranged[key]["Large Responsive"])
            zips["Large Responsive"].append(num_l)
            num_s = len(final_cell_classifications_arranged[key]["Small Responsive"])
            zips["Small Responsive"].append(num_s)
            num_dual = len(final_cell_classifications_arranged[key]["Dual Responsive"])
            zips["Dual Responsive"].append(num_dual)
            num_non = len(final_cell_classifications_arranged[key]["Non-Responsive"])
            zips["Non-Responsive"].append(num_non)

        l_zip = zip(zips["Small Responsive"], zips["Dual Responsive"], zips["Non-Responsive"])
        s_zip = zip(zips["Dual Responsive"], zips["Non-Responsive"])

        # will be top
        bottom_y_l = [x + y + z for (x, y, z) in l_zip]
        bottom_y_s = [x + y for (x, y) in s_zip]
        bottom_y_dual = zips["Non-Responsive"]
        # will be botttom
        bottom_y_non = 0

        ax.bar(x_labels, zips["Non-Responsive"], width, bottom=bottom_y_non, label="Non-Responsive", color="dimgrey")
        ax.bar(x_labels, zips["Dual Responsive"], width, bottom=bottom_y_dual, label="Dual Responsive", color="darkorchid")
        ax.bar(x_labels, zips["Small Responsive"], width, bottom=bottom_y_s, label="Small Responsive", color="indianred")
        ax.bar(x_labels, zips["Large Responsive"], width, bottom=bottom_y_l, label="Large Responsive", color="royalblue")

        ax.set_ylabel("# Cells")
        ax.set_xlabel("Block")
        ax.set_title(f"{session}: Identity Proportions of Shock Non-Responsive Cells Across Blocks")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2,),
            fancybox=True, shadow=True, ncol=4, borderpad=0.3, fontsize="small")
        
        if shock_resp == True:
            label = "respshock"
        else:
            label = "nonrespshock"

        plot_path = os.path.join(dst, f"stacked_plot_{label}_{session}.png")
        plt.savefig(plot_path)



if __name__ == "__main__":
    main()
