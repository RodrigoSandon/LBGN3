import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from matplotlib.pyplot import figure

""" 
Goal:
    - To track cells across sessions
    - The priority will be on Pre-RDT RM, RDT D1, D2, D3
    - Make a graph that displays thefrequency of the number of aligned sessions out of all of these
    - And a separate graph including all session types, not just the ones listed above

"""
def find_paths_endswith(root_path, endswith) -> list:

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )

    return files


def main():


    ROOT = r"/media/rory/Padlock_DT/BLA_Analysis/"
    batch_names = ["PTP_Inscopix_#1", "PTP_Inscopix_#3", "PTP_Inscopix_#4", "PTP_Inscopix_#5"]

    master_d_days_tracked_frequency = {}
    num_mice = 0

    for folder_name in batch_names:
        BATCH_ROOT = os.path.join(ROOT, folder_name)
        mouse_paths = [
            os.path.join(BATCH_ROOT, dir)
            for dir in os.listdir(BATCH_ROOT)
            if os.path.isdir(os.path.join(BATCH_ROOT, dir))
            and dir.startswith("BLA")
        ]
        for mouse_path in mouse_paths:
            print("CURRENT MOUSE PATH: ", mouse_path)
            mouse = mouse_path.split("/")[6]
            num_mice += 1

            try:
                mouse_longreg_csv_file = f"{mouse_path}/longreg_results_preprocessed.csv"
                print(mouse_longreg_csv_file)
                df_longreg = pd.read_csv(mouse_longreg_csv_file)
                prev_cell = None

                d_cell_days_tracked = {}

                for index, row in df_longreg.iterrows():
                    curr_cell = df_longreg.iloc[index, df_longreg.columns.get_loc("global_cell_index")]
                    curr_session = df_longreg.iloc[index, df_longreg.columns.get_loc("session_name")]
                    ############## MODIFY IF YOUR ONLY FOCUSED ON TRACKING CERTAIN SESSION TYPES ##############
                    # only bothering to add if it's either of these session
                    focused_sessions = ["Pre-RDT RM", "RDT D1"]

                    if curr_session in focused_sessions:
                    ############## MODIFY IF YOUR ONLY FOCUSED ON TRACKING CERTAIN SESSION TYPES ##############
                        if curr_cell == prev_cell:
                            d_cell_days_tracked[curr_cell] += 1
                        elif curr_cell != prev_cell:
                            # check if curr cell exists already, if so, add, if not =1
                            prev_cell = curr_cell
                            if curr_cell in d_cell_days_tracked:
                                d_cell_days_tracked[curr_cell] += 1
                            else:
                                d_cell_days_tracked[curr_cell] = 1
                    
                #print(d_cell_days_tracked)
                # get frequency of how many days tracked
                d_days_tracked_frequency = {}

                for key, value in d_cell_days_tracked.items():
                    if value in d_days_tracked_frequency: # if it already exists, add one
                        d_days_tracked_frequency[value] += 1
                    else: # doesn't exist
                        d_days_tracked_frequency[value] = 1
                #print(d_days_tracked_frequency)

                # now add this to the master
                for key, value in d_days_tracked_frequency.items():
                    if key in master_d_days_tracked_frequency:
                        master_d_days_tracked_frequency[key] += value
                    else:
                        master_d_days_tracked_frequency[key] = value

            except FileNotFoundError as e:
                print(f"{mouse} does not have long reg results!")
                pass

    fig, ax = plt.subplots()
    fig.set_figheight(4)
    fig.set_figwidth(6)

    od = collections.OrderedDict(sorted(master_d_days_tracked_frequency.items()))
    print(f"NUMBER OF MICE: {num_mice}")
    print(od)

    x = [int(i) for i in list(od.keys())]
    print(x)
    y = list(od.values())
    ax.bar(x,y, width=0.2, color=(0.2, 0.4, 0.6, 0.6))
    
    print("y: ", y)
    for index, value in enumerate(y):
        ax.text(value, index,str(value))
    rects = ax.patches

    # Make some labels.
    labels = [f"{i}" for i in y]

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height - 1, label, ha="center", va="bottom"
        )
    """every_nth = 2
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    for n, label in enumerate(ax.xaxis.get_major_ticks()):
        if n % every_nth != 0:
            label.set_visible(False)"""
    ax.set_xticks([1,2])
    ax.set_xticklabels(["Pre-RDT RM","RDT D1"])
    ax.set_title("Frequency of Days Neurons are Tracked")
    ax.set_xlabel("Day(s)")
    ax.set_ylabel("Frequency")
    plt.show()


if __name__ == "__main__":
    main()