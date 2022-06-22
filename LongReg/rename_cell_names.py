import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

""" 
Goal:
    - To rename accepted cells to their original names so that we're able to proceed with longreg organization.

"""
def find_paths_endswith(root_path, endswith) -> list:

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )

    return files


def main():


    ROOT = r"/media/rory/Padlock_DT/BLA_Analysis/"
    batch_names = ["PTP_Inscopix_#1", "PTP_Inscopix_#3", "PTP_Inscopix_#4", "PTP_Inscopix_#5"]

    for folder_name in batch_names:
        BATCH_ROOT = os.path.join(ROOT, folder_name)
        mouse_paths = [
            os.path.join(BATCH_ROOT, dir)
            for dir in os.listdir(BATCH_ROOT)
            if os.path.isdir(os.path.join(BATCH_ROOT, dir))
            and dir.startswith("BLA")
        ]
        for mouse_path in mouse_paths:
            mouse = mouse_path.split("/")[6]
            print("CURRENT MOUSE: ", mouse)

            #for each mouse, find the dff_traces.csv and the dff_traces_preprocessed.csv
            raw_dff_traces = find_paths_endswith(mouse_path, "dff_traces.csv")
            preprocessed_dff_traces = [i.replace("dff_traces.csv", "dff_traces_preprocessed.csv") for i in raw_dff_traces]
            # the two lists match with one another

            for idx in range(len(raw_dff_traces)):
                try:
                    d = {}
                    out_path = raw_dff_traces[idx].replace("dff_traces.csv", "naming_change_record.csv")

                    print(f"CURR PATH: {raw_dff_traces[idx]}")
                    raw_df = pd.read_csv(raw_dff_traces[idx])
                    raw_df = raw_df.iloc[:, 1:]
                    og_names = []

                    preprocessed_df = pd.read_csv(preprocessed_dff_traces[idx])
                    preprocessed_df = preprocessed_df.iloc[:, 1:]
                    new_names = list(preprocessed_df.columns)
                    
                    for col in list(raw_df.columns):
                        status = raw_df.iloc[0, raw_df.columns.get_loc(col)]
                        #print(status)
                        if status == " accepted":
                            og_names.append(col)

                    #print(og_names)
                    #print(new_names)
                    d["Old Names"] = og_names
                    d["New Names"] = new_names

                    new_df = pd.DataFrame.from_dict(d)

                    new_df.to_csv(out_path, index=False)
                
                except Exception as e:
                    print(e)
                    pass

if __name__ == "__main__":
    main()