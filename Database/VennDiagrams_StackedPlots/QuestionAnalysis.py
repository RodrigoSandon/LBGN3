import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import os


def stacked_barplot_chain(list_1, list_2, labels, dst):

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

    os.chdir("/home/rory/Rodrigo/Database")

    # Where to store results
    dst = r"/media/rory/Padlock_DT/BLA_Analysis/VennDiagrams/AutoStackedBars"

    # session name : number of cells in session
    session_types = {
        "RDT_D1": 99,
        "RDT_D2": 106,
        "RDT_D3": 39,
    }

    param_to_compare = {
        "Shock": ["True", "False"],
        "Block_Reward_Size": [
            "1dot0_Large",
            "2dot0_Large",
            "3dot0_Large",
            "1dot0_Small",
            "2dot0_Small",
            "3dot0_Small",
        ],
    }

    for param in param_to_compare["Shock"]:
        for param2 in param_to_compare["Block_Reward_Size"]:

            subevent_chain = [
                f"Shock_Ocurred_Choice_Time_s_{param}",
                f"Block_Reward_Size_Choice_Time_s_{param2}",
            ]

            chain_subevent_labels = [
                "Shock_Ocurred_Choice_Time_s_True",
                f"Shock_Ocurred_{param}_Reward_Size_Block__{param}",
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
                        session_df,
                        c,
                        db,
                        session,
                        analysis,
                        "Choice_Time",
                        subevent_chain,
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
