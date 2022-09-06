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

def ptp_autoencoder(mouse_tolower: str) -> str:
    
    mouse_tolower = mouse_tolower.lower()
    d = {
        "PTP_Inscopix_#1": ["bla-insc-1", "bla-insc-2", "bla-insc-3"],
        "PTP_Inscopix_#3": ["bla-insc-5", "bla-insc-6", "bla-insc-7"],
        "PTP_Inscopix_#4": ["bla-insc-8", "bla-insc-9", "bla-insc-11", "bla-insc-13"],
        "PTP_Inscopix_#5": ["bla-insc-14", "bla-insc-15", "bla-insc-16", "bla-insc-18", "bla-insc-19"]
    }

    for key in d.keys():
        if mouse_tolower in d[key]:
            return key

def possible_intermediate(ptp, session_dir):
    # there is an intermediate
    res = ""
    if "PTP_Inscopix_#1" != ptp:
        for dir in os.listdir(session_dir):
            if "BLA" in dir:
                res = dir + "/"
    return res

def main():

    ROOT = r"/media/rory/Padlock_DT/BLA_Analysis/"

    mice = ["BLA-Insc-1",
            "BLA-Insc-6",
            "BLA-Insc-13",
            "BLA-Insc-14",
            "BLA-Insc-15",
            "BLA-Insc-16"]

    sessions = ["Pre-RDT RM", "RDT D1"]

    for mouse in mice:
        print("CURR MOUSE", mouse)
        ptp = ptp_autoencoder(mouse.lower())

        for session in sessions:

            session_dir = f"{ROOT}/{ptp}/{mouse}/{session}"

            intermediate_base = possible_intermediate(ptp, session_dir)

            raw_dff_traces = f"{ROOT}/{ptp}/{mouse}/{session}/{intermediate_base}dff_traces.csv"
            preprocessed_dff_traces = f"{ROOT}/{ptp}/{mouse}/{session}/{intermediate_base}dff_traces_preprocessed.csv"
                
            d = {}
            out_path = raw_dff_traces.replace("dff_traces.csv", "naming_change_record.csv")
            print(out_path)

            print(f"CURR: {mouse} {session}")
            raw_df = pd.read_csv(raw_dff_traces)
            raw_df = raw_df.iloc[:, 1:]
            og_names = []

            preprocessed_df = pd.read_csv(preprocessed_dff_traces)
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
                
                

if __name__ == "__main__":
    main()