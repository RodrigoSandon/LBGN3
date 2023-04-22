import pandas as pd
import os, glob
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import font_manager
import seaborn as sns


def find_paths(root_path: str, end: str) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, "**", end), recursive=True,
    )
    return files

def get_number_of_tracked_cells_longreg_isx(path, first_session, second_session) -> int:
    cells_tracked = 0

    df = pd.read_csv(path)
    print(df.head())
    session_names = list(df["session_name"])
    global_cells = list(df["global_cell_index"])

    for idx, session_name in enumerate(session_names):
        if session_name == first_session:
            # to check if first and second session found within same glob cell #
            curr_global_cell = global_cells[idx]
            next_global_cell = global_cells[idx + 1]
            # to check if first and second session found within same glob cell #
            next_session_name = session_names[idx + 1]
            if next_session_name == second_session and curr_global_cell == next_global_cell:
                # this means the cell was in fact tracked for first and second sessions
                cells_tracked += 1
    
    return cells_tracked

def get_proportion_of_tracked_cells_longreg_isx(path, first_session, second_session) -> int:
    cells_tracked = 0

    df = pd.read_csv(path)
    print(df.head())
    session_names = list(df["session_name"])
    global_cells = list(df["global_cell_index"])

    #remove last two subdir from path
    root_dir = os.path.dirname(path)
    print("ROOT DIR:", root_dir)
    max_total_cells_possible_to_track = find_min_columns_v2(root_dir)
    print("max_total_cells_possible_to_track: ", max_total_cells_possible_to_track)
    #total_cells = 

    for idx, session_name in enumerate(session_names):
        if session_name == first_session:
            # to check if first and second session found within same glob cell #
            curr_global_cell = global_cells[idx]
            next_global_cell = global_cells[idx + 1]
            # to check if first and second session found within same glob cell #
            next_session_name = session_names[idx + 1]
            if next_session_name == second_session and curr_global_cell == next_global_cell:
                # this means the cell was in fact tracked for first and second sessions
                cells_tracked += 1
    print([cells_tracked, max_total_cells_possible_to_track])
    return [cells_tracked, max_total_cells_possible_to_track]

def get_proportion_of_tracked_cells_cellreg(path) -> int:
    
    df = pd.read_csv(path)
    cells_tracked = len(df)
    root_dir = os.path.dirname(path)
    print(root_dir)
    max_total_cells_possible_to_track = find_min_columns_v2(root_dir)
    print("max_total_cells_possible_to_track: ", max_total_cells_possible_to_track)

    print([cells_tracked, max_total_cells_possible_to_track])
    return [cells_tracked, max_total_cells_possible_to_track]

def two_line_plot(d_1,d_1_name, d_2, d_2_name):
    fig, ax = plt.subplots()

    ax.plot(list(range(0, len(list(d_1.values())))),list(d_1.values()), label=d_1_name)
    for i,j in zip(list(range(0, len(list(d_1.values())))),list(d_1.values())):
        ax.annotate(str(j),xy=(i,j))
    ax.plot(list(range(0, len(list(d_2.values())))),list(d_2.values()), label=d_2_name)
    for i,j in zip(list(range(0, len(list(d_2.values())))),list(d_2.values())):
        ax.annotate(str(j),xy=(i,j))

    # Be sure to only pick integer tick locations.
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.set_xticks(range(0, len(list(d_1.values()))))
    ax.set_xticklabels(list(d_1.keys()), rotation=45, fontsize=7)
    ax.set_title("Comparing ISX and CellReg Longitudial Registration Pre-RDT RM & RDT D1")
    ax.set_ylabel("Number of Cells Tracked")
    ax.set_xlabel("Mouse")
    plt.legend(loc='upper left')
    fig.set_size_inches(12, 8)
    fig.savefig("/media/rory/Padlock_DT/Prezis/012723/comparing_isx_cellreg_longreg_prerdtrm_rdtd1.png")

def two_line_plot_v2(d_1, d_1_name, d_2, d_2_name):
    fig, ax = plt.subplots()

    print(d_1.values())
    
    d_1_values = [x[0] / x[1] if isinstance(x, list) else x for x in d_1.values()]

    d_2_values = [x[0] / x[1] if isinstance(x, list) else x for x in d_2.values()]

    ax.plot(list(range(0, len(d_1_values))), d_1_values, label=d_1_name)
    for i, j in zip(list(range(0, len(d_1_values))), d_1.values()):
        try:
            ax.annotate(f"{j[0]}/{j[1]}", xy=(i, j[0] / j[1]))
        except Exception as e:
            ax.annotate(f"{j}", xy=(i, j))

    ax.plot(list(range(0, len(d_2_values))), d_2_values, label=d_2_name)
    for i, j in zip(list(range(0, len(d_2_values))), d_2.values()):
        try:
            ax.annotate(f"{j[0]}/{j[1]}", xy=(i, j[0] / j[1]))
        except Exception as e:
            ax.annotate(f"{j}", xy=(i, j))
    # Be sure to only pick integer tick locations.
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.set_xticks(range(0, len(d_1_values)))
    ax.set_xticklabels(list(d_1.keys()), rotation=45, fontsize=7)
    ax.set_title("Comparing ISX and CellReg Longitudial Registration Pre-RDT RM & RDT D1")
    ax.set_ylabel("Number of Cells Tracked")
    ax.set_xlabel("Mouse")
    plt.legend(loc='upper left')
    fig.set_size_inches(12, 8)
    fig.savefig("/media/rory/Padlock_DT/Prezis/012723/comparing_isx_cellreg_longreg_prerdtrm_rdtd1_proportions.png")

def two_line_plot_v3(d_1, d_1_name, d_2, d_2_name):
    sns.set(style="whitegrid")
    
    fig, ax = plt.subplots()

    print(d_1.values())
    
    d_1_values = [x[0] / x[1] if isinstance(x, list) else x for x in d_1.values()]
    d_2_values = [x[0] / x[1] if isinstance(x, list) else x for x in d_2.values()]

    ax.plot(list(range(0, len(d_1_values))), d_1_values, label=d_1_name, linestyle="-", marker="o", markersize=5, linewidth=2, color='blue')
    for i, j in zip(list(range(0, len(d_1_values))), d_1.values()):
        try:
            ax.annotate(f"{j[0]}/{j[1]}", xy=(i, j[0] / j[1]), fontsize=9, ha='center', va='bottom')
        except Exception as e:
            ax.annotate(f"{j}", xy=(i, j), fontsize=9, ha='center', va='bottom')

    ax.plot(list(range(0, len(d_2_values))), d_2_values, label=d_2_name, linestyle="--", marker="s", markersize=5, linewidth=2, color='red')
    for i, j in zip(list(range(0, len(d_2_values))), d_2.values()):
        try:
            ax.annotate(f"{j[0]}/{j[1]}", xy=(i, j[0] / j[1]), fontsize=9, ha='center', va='bottom')
        except Exception as e:
            ax.annotate(f"{j}", xy=(i, j), fontsize=9, ha='center', va='bottom')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.set_xticks(range(0, len(d_1_values)))
    labels = [i.replace("BLA Insc-", "") for i in list(d_1.keys())]
    ax.set_xticklabels(labels, fontsize=20, fontweight='bold')
    #ax.tick_params(axis='both', labelsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=20)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_manager.FontProperties(size=20, weight='bold'))

    ax.set_title("Comparing ISX and CellReg Longitudial Registration Pre-RDT RM & RDT D1", fontsize=14)
    #ax.set_ylabel("Number of Cells Tracked", fontsize=12)
    #ax.set_xlabel("Mouse", fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    fig.set_size_inches(12, 8)
    fig.savefig("/media/rory/Padlock_DT/Prezis/012723/comparing_isx_cellreg_longreg_prerdtrm_rdtd1_proportions.png")

def two_bar_plot(d_1,d_1_name, d_2, d_2_name):
    d_1_total = sum(list(d_1.values()))
    d_2_total = sum(list(d_2.values()))

    fig, ax = plt.subplots()

    ax.bar([d_1_name, d_2_name],[d_1_total, d_2_total], color = ["#b8d2f2","#F79D9D"])

    rects = ax.patches

    # Make some labels.
    labels = [f"{i}" for i in [d_1_total, d_2_total]]

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height - 5, label, ha="center", va="bottom"
        )

    ax.set_title("Total Cells Tracked for Pre-RDT RM to RDT D1 in All Mice")
    ax.set_ylabel("Number Cells Tracked")
    fig.savefig("/media/rory/Padlock_DT/Prezis/012723/comparing_isx_cellreg_longreg_prerdtrm_rdtd1_total.png")

def find_min_columns(root_directory):
    min_columns = None

    for subdirectory in ['Pre-RDT RM', 'RDT D1']:
        current_path = os.path.join(root_directory, subdirectory)
        
        # Check if the directory exists
        if os.path.exists(current_path):
            files = os.listdir(current_path)
            print("CURRENT PATH: ", current_path)

            # Look for the specified file
            if 'dff_traces_preprocessed.csv' in files:
                file_path = os.path.join(current_path, 'dff_traces_preprocessed.csv')
                df = pd.read_csv(file_path)
                num_columns = len(list(df.columns)) - 1
                print(num_columns)
                # Update min_columns
                if min_columns is None or num_columns < min_columns:
                    min_columns = num_columns

    if min_columns is None:
        min_columns = 1
    return min_columns

def find_min_columns_v2(root_directory):
    min_columns = 0
    max_depth = 2

    for subdirectory in ['Pre-RDT RM', 'RDT D1']:
        current_path = os.path.join(root_directory, subdirectory)

        # Check if the directory exists
        if os.path.exists(current_path):
            for root, dirs, files in os.walk(current_path):
                depth = root[len(current_path) + len(os.path.sep):].count(os.path.sep)

                if depth <= max_depth:
                    # Look for the specified file
                    if 'dff_traces_preprocessed.csv' in files:
                        file_path = os.path.join(root, 'dff_traces_preprocessed.csv')
                        df = pd.read_csv(file_path)
                        num_columns = len(list(df.columns)) - 1

                        # Update min_columns
                        if min_columns == 0 or num_columns < min_columns:
                            min_columns = num_columns
                else:
                    # Stop the search at the current depth
                    dirs[:] = []

    if min_columns == 0:
        min_columns = 1
    return min_columns

# Replace with the actual list of root directories

def main():
    session_types = [
            
            "Pre-RDT RM",
            "RDT D1",
    ]

    mice = [
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-2",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-3",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-5",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-7",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-8",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-9",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-11",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-13",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-14",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-15",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-16",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-18",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-19",
        ]

    mice_isx_longreg_paths = ["/".join([i,"longreg_results_preprocessed.csv"]) for i in mice]
    mice_cellreg_paths = ["/".join([i,"cellreg_Pre-RDT RM_RDT D1.csv"]) for i in mice]

    #### First count number of cells tracked for longreg ####

    # Set up dict #
    mice_isx_longreg_count_d = {
        "BLA-Insc-1" : 0,
        "BLA-Insc-2": 0,
        "BLA-Insc-3": 0,
        "BLA-Insc-5": 0,
        "BLA-Insc-6": 0,
        "BLA-Insc-7": 0,
        "BLA-Insc-8": 0,
        "BLA-Insc-9": 0,
        "BLA-Insc-11": 0,
        "BLA-Insc-13": 0,
        "BLA-Insc-14": 0,
        "BLA-Insc-15": 0,
        "BLA-Insc-16": 0,
        "BLA-Insc-18": 0,
        "BLA-Insc-19": 0,
    }

    # Go through each csv and count

    for path in mice_isx_longreg_paths:
        mouse = path.split("/")[-2]
        try:
            # returns [cells_tracked, max_total_cells_possible_to_track]
            mice_isx_longreg_count_d[mouse] =  get_proportion_of_tracked_cells_longreg_isx(path, session_types[0], session_types[1])
        except FileNotFoundError as e:
            print(e)
            pass
    print("mice_isx_longreg_count_d")    
    print(mice_isx_longreg_count_d)

    #### First count number of cells tracked for cellreg ####

    # Set up dict #
    mice_cellreg_count_d = {
        "BLA-Insc-1" : 0,
        "BLA-Insc-2": 0,
        "BLA-Insc-3": 0,
        "BLA-Insc-5": 0,
        "BLA-Insc-6": 0,
        "BLA-Insc-7": 0,
        "BLA-Insc-8": 0,
        "BLA-Insc-9": 0,
        "BLA-Insc-11": 0,
        "BLA-Insc-13": 0,
        "BLA-Insc-14": 0,
        "BLA-Insc-15": 0,
        "BLA-Insc-16": 0,
        "BLA-Insc-18": 0,
        "BLA-Insc-19": 0,
    }

    # Go through each csv and count

    for path in mice_cellreg_paths:
        mouse = path.split("/")[-2]
        try:
            mice_cellreg_count_d[mouse] =  get_proportion_of_tracked_cells_cellreg(path)
        except FileNotFoundError as e:
            print(e)
            pass
    print("mice_cellreg_count_d")    
    print(mice_cellreg_count_d)

    two_line_plot_v3(mice_isx_longreg_count_d, "isx_longreg(70%)", mice_cellreg_count_d, "cellreg(default)")
    two_bar_plot(mice_isx_longreg_count_d, "isx_longreg(70%)", mice_cellreg_count_d, "cellreg(default)")
    

if __name__ == "__main__":
    main()