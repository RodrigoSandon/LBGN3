import os
import glob

def find_paths_endswith_recur_n_norecur(root_path, endswith) -> list:

    files_2 = []

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )

    # now if recursive is empty, then check in just the current tree
    if len(files) == 0:
        print("here") # if prints that means this code doesn't acc check curr tree

        for i in os.listdir(root_path):
            if endswith in i:   
                files_2.append(os.path.join(root_path, i))
        
        print(files_2)
        return files_2
    
    print(files)
    return files

root_path = "/media/rory/Padlock_DT/TestingEnvironment"
endswith = ".txt"
find_paths_endswith_recur_n_norecur(root_path, endswith)