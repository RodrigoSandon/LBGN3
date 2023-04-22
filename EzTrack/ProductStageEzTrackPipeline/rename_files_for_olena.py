import os, glob, re


def find_paths(root_path, endswith: str):
    files = glob.glob(
        os.path.join(root_path, "**", f"*{endswith}"), recursive=True,
    )
    return files

def rename_files(root_dir, dry_run):
    files = find_paths(root_dir, ".avi")

    #print(files)
    for file_path in files:
        #print(file_name)
        #match = re.search(r"hSyn-AS-Gi-\d+_\d+-\d+-\d+_Cam\d.avi", file_name)
        file_name = os.path.basename(file_path)
        start = "hSyn-AS-Gi-"
        to_replace = r"\d+_\d+"
        end = r"-\d+-\d+_Cam\d.avi"
        match_to_replace = re.search(to_replace, file_name)
        match_end = re.search(end, file_name)

        if match_to_replace:
            if "Cam1" in file_name:
                number = match_to_replace.group().split("_")[0]
                new_name = f"{start}{number}{match_end.group()}"
            elif "Cam2" in file_name:
                number = match_to_replace.group().split("_")[1]
                new_name = f"{start}{number}{match_end.group()}"
            else:
                continue  # skip if neither "Cam1" nor "Cam2" in filename
            new_path = file_path.replace(file_name, new_name)
            if dry_run == True:
                #print(f"Old Name: {file_name} New Name: {new_name}")
                print(f"Old Name: {file_path} New Name: {new_path}")
            else:
                
                os.rename(file_path, new_path)
                print(f"Old Name: {file_path} New Name: {new_path}")

ROOT = "/media/rory/Padlock_DT/Fear_Conditioning_Control/Olena_Group/"

rename_files(ROOT, dry_run=False)
