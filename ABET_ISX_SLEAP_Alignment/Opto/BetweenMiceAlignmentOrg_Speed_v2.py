import pandas as pd
import numpy as np
import os, glob, shutil
from pathlib import Path

""" Finds path with wildcard intermediate paths and wildcard for whatever
    comes before filename. """
def find_paths_endswith(root, endswith) -> list:

    files = glob.glob(os.path.join(root, "**", f"*{endswith}"), recursive=True)

    return files

def find_paths_no_middle_endswith(root, endswith) -> list:

    files = []

    for i in os.listdir(root):
        if i.endswith(endswith):
            files.append(i)

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

def verify_sleap_file(animal_id, filepath_1, filepath_2, session_type, not_session_type) -> str:

    if (session_type.lower() in filepath_1.lower()) and (not_session_type.lower() not in filepath_1.lower()) or (not_session_type.lower() not in filepath_1.lower()):
        #print("File is going where it's supposed to go.")
        return "OK"
    else:
        print(f"{animal_id} is NOT going where it's supposed to go:")
        print(f"souce: {filepath_1}")
        print(f"dst: {filepath_2}")
        return "BAD"


def main():
    # WE ARE INCLUDING FILES FOUND IN BOTH BORIS & BORIS_MERGE
    ROOT = r"/media/rory/RDT VIDS"
    DST = r"/media/rory/Padlock_DT/Opto_Speed_Analysis/Analysis_2"

    ROOT_1 = r"/media/rory/RDT VIDS/BORIS_merge"
    ROOT_2 = r"/media/rory/RDT VIDS/BORIS"
    # So the thing is, some boris_merge files missing their .slp files, so find in batch2
    # should be only search for slp files im pretty sure
    # need to search bc slp files labels tell me whether it came from choice or not
    ROOT_3 = r"/media/rory/RDT VIDS/BATCH_2"

    """Looks thru different folders to find mice in each of the rows - there are many idosyncrazies
    in the way the data is stored and structured that make this code complicated, specially
    with it's verification - very specific to the situation, dont have it organized like this next time"""
    circuits_d = {
        "BLA_NAcShell" : "/media/rory/Padlock_DT/Opto_Speed_Analysis/Opto Data Info - BLA-NAcShell.csv",
        "vHPC_NAcShell" : "/media/rory/Padlock_DT/Opto_Speed_Analysis/Opto Data Info - vHPC-NAcShell.csv",
        "vmPFC_NAcShell" : "/media/rory/Padlock_DT/Opto_Speed_Analysis/Opto Data Info - vmPFC-NAcShell.csv"
    }
    session_type = "Choice"
    session_type_lower = "choice"
    not_session_type = "Outcome"
    not_session_type_lower = "outcome"
    #there are missing labels for each possibly, maybe it should be different after manually renaming things
    combos = ["Block_Trial_Type_Start_Time_(s)"]
    node_type = "body"

    num_sleap_data_files_for_existing_mice_not_found = 0
    
    for circuit, organizer_path in circuits_d.items():
        print("Current circuit: ", circuit)
        df_organizer = pd.read_csv(organizer_path)
        num_rows = len(df_organizer)
        # We're interested in columns: Animal ID, Session, Treatment
        
        for row in range(0,num_rows):
    
            animal_id = modify_animal_id(df_organizer.loc[row, "Animal ID"])
            
            treatment = df_organizer.loc[row, "Treatment"]
            sleap_data_file : str
            alignment_folder: str
            slp_file: str

            print("Current mouse: ", animal_id)

            # find example: f"{root_folder_X}/{animal_id}/**/{animal_id}_{session_type_lower}_{node_type}_sleap_data.csv"
            # dst example: f"{root_dst}/{circuit}/{treatment}/{session_type}/{animal_id}/{node_type}/{animal_id}_{session_type_lower}_{node_type}_sleap_data.csv"
    
            
            try:
                sleap_data_file = find_paths_endswith(f"{ROOT_1}/{animal_id}", f"{animal_id}_{session_type_lower}_{node_type}_sleap_data.csv")

                if sleap_data_file:
                    # non empty
                    # either choice in there or notchoice not in there
                    sleap_data_file = sleap_data_file[0]
                    slp_file = find_paths_no_middle_endswith(f"{ROOT_1}/{animal_id}", ".slp")
                    if slp_file:
                        #if there are more than two, get the one with notchoice not in it
                        slp_file = [i for i in slp_file if not_session_type.lower() not in i.lower()]

                        if len(slp_file) == 0:
                            print(f"{animal_id} isn't {session_type} video predicted.")
                            continue
                        else:
                            slp_file = slp_file[0]
                    else:
                        slp_file = find_paths_endswith(f"{ROOT_1}/{animal_id}", ".slp")
                        #print(slp_file)

                        if slp_file:
                            #if there are more than two, get the one with notchoice not in it
                            slp_file = [i for i in slp_file if not_session_type.lower() not in i.lower()]

                            if len(slp_file) == 0:
                                print(f"{animal_id} isn't {session_type} video predicted.")
                                continue
                            else:
                                slp_file = slp_file[0]
                        else:
                            slp_file = find_paths_endswith(f"{ROOT_3}/{animal_id}", ".slp")

                            #if there are more than two, get the one with notchoice not in it
                            slp_file = [i for i in slp_file if not_session_type.lower() not in i.lower()]

                            if len(slp_file) == 0:
                                print(f"{animal_id} isn't {session_type} video predicted.")
                                continue
                            else:
                                slp_file = slp_file[0]

                    if session_type.lower() in sleap_data_file.lower() or not_session_type.lower() not in sleap_data_file.lower():
                        alignment_folder = directory_find(f"{ROOT_1}/{animal_id}/",f"{animal_id}_{session_type_lower}_AlignmentData")
                    else:
                        # either choice not in there or notchoice in there, so look in other folder
                        sleap_data_file = find_paths_endswith(f"{ROOT_2}/{animal_id}", f"{animal_id}_{session_type_lower}_{node_type}_sleap_data.csv")
                        sleap_data_file = sleap_data_file[0]
                        
                        slp_file = find_paths_no_middle_endswith(f"{ROOT_2}/{animal_id}", ".slp")
                        #print(slp_file)

                        if slp_file:
                            #if there are more than two, get the one with notchoice not in it
                            slp_file = [i for i in slp_file if not_session_type.lower() not in i.lower()]

                            if len(slp_file) == 0:
                                print(f"{animal_id} isn't {session_type} video predicted.")
                                continue
                            else:
                                slp_file = slp_file[0]
                        else:
                            slp_file = find_paths_endswith(f"{ROOT_2}/{animal_id}", ".slp")
                            #print(slp_file)

                            if slp_file:
                                #if there are more than two, get the one with notchoice not in it
                                slp_file = [i for i in slp_file if not_session_type.lower() not in i.lower()]

                                if len(slp_file) == 0:
                                    print(f"{animal_id} isn't {session_type} video predicted.")
                                    continue
                                else:
                                    slp_file = slp_file[0]
                            else:
                                slp_file = find_paths_endswith(f"{ROOT_3}/{animal_id}", ".slp")

                                #if there are more than two, get the one with notchoice not in it
                                slp_file = [i for i in slp_file if not_session_type.lower() not in i.lower()]

                                if len(slp_file) == 0:
                                    print(f"{animal_id} isn't {session_type} video predicted.")
                                    continue
                                else:
                                    slp_file = slp_file[0]


                        if sleap_data_file:
                            # if non empty, check to see if meet criteria again
                            if session_type.lower() in sleap_data_file.lower() or not_session_type.lower() not in sleap_data_file.lower():
                                alignment_folder = directory_find(f"{ROOT_2}/{animal_id}/",f"{animal_id}_{session_type_lower}_AlignmentData")
                        else:
                            # if empty after second look
                            print(f"{animal_id} does not have a {session_type} {animal_id}_{session_type_lower}_{node_type}_sleap_data.csv")
                            num_sleap_data_files_for_existing_mice_not_found += 1
                            continue
                else:
                    # empty, so look in other folder
                    sleap_data_file = find_paths_endswith(f"{ROOT_2}/{animal_id}", f"{animal_id}_{session_type_lower}_{node_type}_sleap_data.csv")
                    sleap_data_file = sleap_data_file[0]
                    
                    slp_file = find_paths_no_middle_endswith(f"{ROOT_2}/{animal_id}", ".slp")
                    #print(slp_file)

                    if slp_file:
                        #if there are more than two, get the one with notchoice not in it
                        slp_file = [i for i in slp_file if not_session_type.lower() not in i.lower()]

                        if len(slp_file) == 0:
                            print(f"{animal_id} isn't {session_type} video predicted.")
                            continue
                        else:
                            slp_file = slp_file[0]
                    else:
                        slp_file = find_paths_endswith(f"{ROOT_2}/{animal_id}", ".slp")
                        #print(slp_file)

                        if slp_file:
                            #if there are more than two, get the one with notchoice not in it
                            slp_file = [i for i in slp_file if not_session_type.lower() not in i.lower()]

                            if len(slp_file) == 0:
                                print(f"{animal_id} isn't {session_type} video predicted.")
                                continue
                            else:
                                slp_file = slp_file[0]
                        else:
                            slp_file = find_paths_endswith(f"{ROOT_3}/{animal_id}", ".slp")

                            #if there are more than two, get the one with notchoice not in it
                            slp_file = [i for i in slp_file if not_session_type.lower() not in i.lower()]

                            if len(slp_file) == 0:
                                print(f"{animal_id} isn't {session_type} video predicted.")
                                continue
                            else:
                                slp_file = slp_file[0]


                    if sleap_data_file:
                        # if non empty, check to see if meet criteria again
                        if session_type.lower() in sleap_data_file.lower() or not_session_type.lower() not in sleap_data_file.lower():
                            alignment_folder = directory_find(f"{ROOT_2}/{animal_id}/",f"{animal_id}_{session_type_lower}_AlignmentData")
                    else:
                        # if empty after second look
                        print(f"{animal_id} does not have a {session_type} {animal_id}_{session_type_lower}_{node_type}_sleap_data.csv")
                        num_sleap_data_files_for_existing_mice_not_found += 1
                        continue
                
                # Save to csv
                df_organizer.to_csv(organizer_path, index=False)

                DST_NEW_PATH = f"{DST}/{circuit}/{treatment}/{session_type}/{animal_id}/{node_type}/"
                dst_file = f"{DST_NEW_PATH}/{animal_id}_{session_type_lower}_{node_type}_sleap_data.csv"

                result = verify_sleap_file(animal_id, slp_file, dst_file, session_type, not_session_type)

                if result == "OK":
                    os.makedirs(DST_NEW_PATH, exist_ok=True)
                    
                    #print("ALIGNMENT FOLDER:")
                    #print(f"{DST_NEW_PATH}/AlignmentData")
                    #print("dst: ", f"")
                    shutil.copytree(alignment_folder, f"{DST_NEW_PATH}/{animal_id}_{session_type_lower}_AlignmentData")

                    shutil.copyfile(sleap_data_file, dst_file)

            except (FileExistsError, TypeError, AttributeError, IndexError) as e:
                print(e)
                pass

    print("num_sleap_data_files_for_existing_mice_not_found: ", num_sleap_data_files_for_existing_mice_not_found)



if __name__ == "__main__":
    main()