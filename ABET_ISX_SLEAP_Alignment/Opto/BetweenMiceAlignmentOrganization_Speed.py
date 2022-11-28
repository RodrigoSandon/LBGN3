import pandas as pd
import numpy as np
import os, glob, shutil
from pathlib import Path

""" Finds path with wildcard intermediate paths and wildcard for whatever
    comes before filename. """
def find_paths_endswith(root, endswith) -> list:

    files = glob.glob(os.path.join(root, "**", f"{endswith}"), recursive=True)

    return files

def find_paths_middle_filename(root, middle, filename) -> list:

    files = glob.glob(os.path.join(root, "**", middle, "**", filename), recursive=True)

    return files

def directory_find(root, atom):
    for path, dirs, files in os.walk(root):
        if atom in dirs:
            return os.path.join(path, atom)

def find_paths_no_middle_contains(root, contains):

    result = []
    for file in os.listdir(root):
        file = file.lower()
        if contains in file:
            result.append(file)

    return result

def determine_session_type(mystr: str) -> str:
    mystr = mystr.lower()

    if ("choice" in mystr) and ("rm" not in mystr) :
        return "Choice"

    return "Outcome"

def modify_animal_id(mystr: str) ->  str:

    if ("RRD" not in mystr) and ("rrd" not in mystr):
        # must not contain neither of these for these to happen
        mystr = "RRD" + mystr

    return mystr

def verify_match(filepath_1, filepath_2):
    filepath_1 = filepath_1.lower()
    filepath_2 = filepath_2.lower()

    if ("choice" in filepath_1 and "choice" in filepath_2) or ("outcome" in filepath_1 and "outcome" in filepath_2):
        #print("File is going where it's supposed to go.")
        pass
    else:
        print("File is NOT going where it's supposed to go.")



def main():
    # WE ARE INCLUDING FILES FOUND IN BOTH BORIS & BORIS_MERGE (WHICH INCLUDES BATCH_2)
    # The data that we want to categorize are in two different folders
    root_folder = r"/media/rory/RDT VIDS"
    root_dst = "/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis"
    root_for_missing = r"/media/rory/RDT VIDS/BORIS_merge"
    drive_to_look_in = r"/media/rory/Risk Videos/"
    # dst example: f"{root_dst}/{circuit}/{treatment}/{session_type}/{animal_id}/{node_type}/{node_type}_sleap_data.csv""
    # session_type: choice or outcomes or DNE, node_type: body for now
    # circuit : filepath_to_organizer

    root_folders = ["BORIS", "BORIS_merge"]
    circuits_d = {
        "BLA_NAcShell" : "/media/rory/Padlock_DT/Opto_Speed_Analysis/Opto Data Info - BLA-NAcShell.csv",
        "vHPC_NAcShell" : "/media/rory/Padlock_DT/Opto_Speed_Analysis/Opto Data Info - vHPC-NAcShell.csv",
        "vmPFC_NAcShell" : "/media/rory/Padlock_DT/Opto_Speed_Analysis/Opto Data Info - vmPFC-NAcShell.csv"
    }

    treatments = ["eYFP", "ArchT", "mCherry", "ChrimsonR"]
    node_type = "body"
    session_types = ["choice", "outcome"]

    combos_we_care_about = "Block_Trial_Type_Start_Time_(s)"

    to_sleap_process = []

    
    for circuit, organizer_path in circuits_d.items():
        print("Current circuit: ", circuit)
        df_organizer = pd.read_csv(organizer_path)
        num_rows = len(df_organizer)

        # We're interested in columns: Animal ID, Session, Treatment
        
        for row in range(0,num_rows):
    
            animal_id = modify_animal_id(df_organizer.loc[row, "Animal ID"])
            treatment = df_organizer.loc[row, "Treatment"]
            sleap_file : str
            alignment_folder: str

            print("Current mouse: ", animal_id)

            # find example: f"{root_folder_X}/{animal_id}/**/{node_type}_sleap_data.csv"
            # dst example: f"{root_dst}/{circuit}/{treatment}/{session_type}/{animal_id}/{node_type}/{node_type}_sleap_data.csv"
            
            #temp = find_paths_middle_filename(f"{root_folder}/BORIS_merge/{animal_id}", combos_we_care_about, filename)
            
            try:
                temp = find_paths_endswith(f"{root_folder}/BORIS_merge/{animal_id}", f"{node_type}_sleap_data.csv")

                if len(temp) == 0:
                    temp = find_paths_endswith(f"{root_folder}/BORIS/{animal_id}", f"{node_type}_sleap_data.csv")

                    if len(temp) == 0:
                        # sleap_file wasnt found, assuming there's a hard drive we can extract it from if missing
                        #print(f"SLEAP file for {animal_id} not found.")

                        
                        if "outcome" not in temp: # only choice for now 9/28/22
                            new_missing_dir = f"{root_for_missing}/{animal_id}/{session_types[0]}"
                            os.makedirs(new_missing_dir, exist_ok=True)

                            # Now need to grab specific videos to copy, don't assume the one's you're looking for will be there, assume all caps?
                            separated_vids_to_merge = find_paths_no_middle_contains(f"{drive_to_look_in}/{animal_id}", ses_type)

                            separated_vids_to_merge = [i for i in separated_vids_to_merge if "outcome" not in i.lower()]
                            

                            if len(separated_vids_to_merge) == 0:
                                df_organizer.loc[row, "Video exists?"] = "F"
                                df_organizer.loc[row, "Merged?"] = "F"
                                df_organizer.loc[row, "Predicted?"] = "F"
                                df_organizer.loc[row, "Averaged?"] = "F"
                                #print(f"{session_types[0]} videos for {animal_id} merging not found.")
                                continue
                            else:
                                to_sleap_process.append(animal_id + f"_{session_types[0]}")

                            for vid in separated_vids_to_merge:
                                df_organizer.loc[row, "Video exists?"] = "T"
                                df_organizer.loc[row, "Merged?"] = "F"
                                df_organizer.loc[row, "Predicted?"] = "F"
                                df_organizer.loc[row, "Averaged?"] = "F"
                                filename = Path(vid).name
                                dst = os.path.join(new_missing_dir, filename)
                                #print(f"Copying {vid} to {dst}")
                                #shutil.copyfile(f"{drive_to_look_in}/{animal_id}/{vid}", dst)
                                # Now do merging + sleap pipeline separately

                    else:
                        # sleap file found in BORIS_merge
                        sleap_file = temp[0]
                        alignment_folder = directory_find(f"{root_folder}/BORIS/{animal_id}/","AlignmentData")

                
                else:
                    sleap_file = temp[0]
                    alignment_folder = directory_find(f"{root_folder}/BORIS_merge/{animal_id}/","AlignmentData")
                
                # if it found the sleap file, this must mean ...
                df_organizer.loc[row, "Video exists?"] = "T"
                df_organizer.loc[row, "Merged?"] = "T"
                df_organizer.loc[row, "Predicted?"] = "T"
                df_organizer.loc[row, "Averaged?"] = "F"
                # Save to csv
                df_organizer.to_csv(organizer_path, index=False)

                #print("SLEAP file found:", sleap_file)
                # Determine session type based on sleap file dir
                #session_type = determine_session_type(sleap_file)
                dst_folder = f"{root_dst}/{circuit}/{treatment}/{session_types[0]}/{animal_id}/{node_type}/"
                dst_file = f"{dst_folder}/{node_type}_sleap_data.csv"

                os.makedirs(dst_folder, exist_ok=True)
                
                #print("ALIGNMENT FOLDER:")
                print(dst_folder)
                #print("dst: ", f"")
                shutil.copytree(alignment_folder, f"{dst_folder}/AlignmentData")

                #print("SLEAP file destination:", dst_file)
                verify_match(sleap_file, dst_file)

                shutil.copyfile(sleap_file, dst_file)
            except (FileExistsError, TypeError) as e:
                print(e)
                pass

    print("Make sure to sleap analyze the following vids next: ", to_sleap_process)



if __name__ == "__main__":
    main()