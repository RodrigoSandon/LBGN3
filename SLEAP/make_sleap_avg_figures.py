import os, glob
import pandas as pd
from typing import List

def find_paths_endswith(root_path, endswith) -> List:

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )

    return files

def main():
    
    ROOT_PATH = "/media/rory/Padlock_DT/BLA_Analysis"
    sessions = ["RDT D1", "RDT D2", "RDT D3"]
    cell = "C01" # doesn't matter which cell you choose, all the same data
    event = "Block_Reward Size_Choice Time (s)"
    subevents = ["(1.0, 'Large')", "(1.0, 'Small')", "(2.0, 'Large')", "(2.0, 'Small')", "(3.0, 'Large')", "(3.0, 'Small')"]
    filename = "avg_unnorm_dff_speed_plot_ready.csv"

    for session in sessions:
        print(session)
        for subevent in subevents:
            print(f"Curr subevent: {subevent}")
            # will vary only in mouse they from
            speed_files = find_paths_endswith(ROOT_PATH, f"{session}/SingleCellAlignmentData/{cell}/{event}/{subevent}/{filename}")

            num_files = len(speed_files)

            summed = None

            for speed_file in speed_files:
                df = pd.read_csv(speed_file)
                speed_list = list(df["Speed (cm/s)"])
                if summed == None:
                    summed = speed_list
                else:
                    zipped = zip(summed, speed_list)
                    try:
                        summed = [x + y for (x,y) in zipped]
                    except TypeError as e:
                        print(speed_file)
                        print(speed_list)

            avg = [i/num_files for i in summed]

            #save this avg in betweenmicealignmentdata
            avg_df = pd.DataFrame(avg, columns=["Avg. Speed (cm/s)"])
            dst = f"{ROOT_PATH}/BetweenMiceAlignmentData/{session}/{event}/{subevent}/avg_unnorm_speed.csv"
            avg_df.to_csv(dst, index=False)
        break
            

if __name__ == "__main__":
    main()