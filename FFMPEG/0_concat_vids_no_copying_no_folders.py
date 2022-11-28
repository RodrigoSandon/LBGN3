import os, glob
import shutil

def find_paths_endswith(root_path, endswith: str):
    files = glob.glob(
        os.path.join(root_path, endswith), recursive=True,
    )
    return files

def find_paths_startswith_endswith(root_path, startwith: str, endswith: str):
    files = glob.glob(
        os.path.join(root_path, "**", f"{startwith}*{endswith}"), recursive=True,
    )
    return files

def find_paths_startswith(root_path, startswith):

    files = glob.glob(
        os.path.join(root_path, "**", "%s*") % (startswith), recursive=True,
    )

    return files

def delete_recursively(root, name_endswith_list):

    for i in name_endswith_list:

        files = glob.glob(os.path.join(root, "**", "*%s") % (i), recursive=True)

        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))

def get_mouse_name(filename: str):
    mouse = filename.split("_")[0]

    return mouse


#RRD44_Risk_D1_0.1_mA_OPTO_ALL_FREE_CHOICE_04162019_3
def main():
    root = r"/media/rory/RDT VIDS/BORIS_merge/BATCH_2"
    dst = r"/media/rory/RDT VIDS/BORIS_merg/BATCH_2"
    
    already_processed = find_paths_startswith_endswith(root, "RRD", "merged.mp4")
    #to know which mice are done, and to update
    already_processed = [i.split("/")[-1].split("_")[0] for i in already_processed]
    print(already_processed)


    for file in os.listdir(root):
        if os.path.isdir(file) == False:
            curr_mouse = get_mouse_name(file)
            #print(already_processed)
            to_process = True

            for mouse_name in already_processed:
                if curr_mouse == mouse_name:
                    to_process = False

            if to_process == True:
                print("CURR FILE:", file)
                already_processed.append(curr_mouse)
                # get sessions only of this mouse
                # make sure it doesnt get for ex rrd167 if your looking for a file that starts with rrd16
                curr_mouse_vid_parts = find_paths_startswith(f"{root}/{curr_mouse}", curr_mouse.lower() + "_")
                print("vids found: ", curr_mouse_vid_parts)

                new_folder_name = os.path.join(root, curr_mouse)
                # print("new folder name", new_folder_name)

                os.makedirs(new_folder_name, exist_ok=True)


                # open new text file
                os.chdir(new_folder_name)  # in dst
                # del old mylist.txt's if any exist
                """delete_recursively(root, "mylist.txt")
                delete_recursively(root, "mylist2.txt")"""

                print("Curr dir:", os.getcwd())


                file = open("mylist.txt", "w")
                name_merged_file = ""
                L = []
                # sort files
                files_sorted = []
                for name in curr_mouse_vid_parts:
                    if name != os.path.join(root, curr_mouse):
                    
                        files_sorted.append(name)

                files_sorted.sort()
                for name in files_sorted:         

                    if " " in name:
                        """os.rename(
                            os.path.join(root, name),
                            os.path.join(root, name.replace(" ", "\ ")),
                        )"""
                        name = name.replace(" ", "\ ")
                    if "(" and ")" in name:
                        os.rename(
                            os.path.join(root, name),
                            os.path.join(
                                root, name.replace("(", "").replace(")", "")
                            ),
                        )
                        name = name.replace("(", "").replace(")", "")
                    # change acc folder name
                    # and change variable name accordingly (change it the same way)
                    # and make sure this is reflected in mylist.txt
                    name_merged_file = name.replace(".MP4", "")
                    # when there is no quotation marks surrounding file path- it's better bc allows for whitespaces
                    name_str = "file %s" % (name) + "\n"
                    print(name_str)
                    L.append(name_str)

                # get file name

                # print(L)
                file.writelines(L)
                file.close()
                # copy this file to dst
                #copyfile("mylist.txt", os.path.join(new_folder_name, "mylist.txt"))
                # where to look for the individual videos
                dst_path = os.path.join(
                    new_folder_name.replace(" ", r"\ "), name_merged_file
                )
                merged_file_path = f"{dst_path}_merged.mp4"
                if name_merged_file != "":
                    cmd = f"ffmpeg -f concat -safe 0 -i mylist.txt -c copy {merged_file_path}"
                    print(cmd)
                    os.system(cmd)

                    """if r"\\" in merged_file_path:
                        merged_file_path = merged_file_path.replace(r"\\","")

                    shutil.move(merged_file_path, os.path.join(new_folder_name, merged_file_path.split("/")[0]))"""


def main2():
    root = r"/media/rory/RDT VIDS/BORIS_merge/BATCH_2"
    dst = r"/media/rory/RDT VIDS/BORIS_merg/BATCH_2"
    
    already_processed_1 = find_paths_startswith_endswith(root, "rrd", "merged.mp4")
    already_processed_2 = find_paths_startswith_endswith(root, "RRD", "merged.mp4")
    #to know which mice are done, and to update
    already_processed_1 = [i.split("/")[-1].split("_")[0] for i in already_processed_1]
    already_processed_2 = [i.split("/")[-1].split("_")[0] for i in already_processed_2]

    already_processed = already_processed_1 + already_processed_2
 
    print("ALREADY PROCESSED")
    print(already_processed)


    for file in os.listdir(root):
        if os.path.isdir(file) == False:
            curr_mouse = get_mouse_name(file)
            #print(already_processed)
            to_process = True

            for mouse_name in already_processed:
                if curr_mouse.lower() == mouse_name:
                    to_process = False

            if to_process == True:
                print("CURR FILE:", file)
                already_processed.append(curr_mouse)
                # get sessions only of this mouse
                # make sure it doesnt get for ex rrd167 if your looking for a file that starts with rrd16
                curr_mouse_vid_parts = find_paths_startswith(f"{root}/{curr_mouse}", curr_mouse.lower()) # may need to add an + '_'
                print("vids found: ", curr_mouse_vid_parts)

                new_folder_name = os.path.join(root, curr_mouse)
                # print("new folder name", new_folder_name)

                os.makedirs(new_folder_name, exist_ok=True)


                # open new text file
                os.chdir(new_folder_name)  # in dst
                # del old mylist.txt's if any exist

                print("Curr dir:", os.getcwd())


                file = open("mylist.txt", "w")
                name_merged_file = ""
                L = []
                # sort files
                files_sorted = []
                for name in curr_mouse_vid_parts:
                    if name != os.path.join(root, curr_mouse):
                    
                        files_sorted.append(name)

                files_sorted.sort()
                for name in files_sorted:         

                    if " " in name:

                        name = name.replace(" ", "\ ")
                    if "(" and ")" in name:

                        name = name.replace("(", "\(").replace(")", "\)")
                    
                    name_merged_file = name.replace(".MP4", "")
                    # when there is no quotation marks surrounding file path- it's better bc allows for whitespaces
                    name_str = "file %s" % (name) + "\n"
                    print(name_str)
                    L.append(name_str)

                # get file name

                # print(L)
                file.writelines(L)
                file.close()
                # copy this file to dst
                #copyfile("mylist.txt", os.path.join(new_folder_name, "mylist.txt"))
                # where to look for the individual videos
                """dst_path = os.path.join(
                    new_folder_name.replace(" ", r"\ "), name_merged_file
                )"""
                dst_path = os.path.join(
                    new_folder_name, name_merged_file
                )
                merged_file_path = f"{dst_path}_merged.mp4"
                if name_merged_file != "":
                    cmd = f"ffmpeg -f concat -safe 0 -i mylist.txt -c copy {merged_file_path}"
                    print(cmd)
                    os.system(cmd)

def process_one():

    # Insert folder of the mouse you want to concat
    root = r"/media/rory/RDT VIDS/BORIS_merge/RRD38"
 
    curr_mouse = root.split("/")[-1]

    # get sessions only of this mouse
    # make sure it doesnt get for ex rrd167 if your looking for a file that starts with rrd16
    curr_mouse_vid_parts = []
    #curr_mouse_vid_parts = find_paths_endswith(root, ".mp4") # may need to add an + '_'
    #print(os.listdir(root))
    for i in os.listdir(root):
        if ".MP4" in i and "merge" not in i:
            curr_mouse_vid_parts.append(os.path.join(root, i))
    print("vids found: ", curr_mouse_vid_parts)

    os.chdir(root)  # in dst
    # del old mylist.txt's if any exist

    print("Curr dir:", os.getcwd())

    # print("new folder name", new_folder_name)

    file = open(f"{root}/mylist.txt", "w")

    name_merged_file = ""
    L = []
    # sort files
    files_sorted = []
    for name in curr_mouse_vid_parts:
        if name != os.path.join(root, curr_mouse):
        
            files_sorted.append(name)

    files_sorted.sort()
    for name in files_sorted:         

        if " " in name:

            name = name.replace(" ", "\ ")
        if "(" and ")" in name:

            name = name.replace("(", "\(").replace(")", "\)")
        
        name_merged_file = name.replace(".MP4", "")
        # when there is no quotation marks surrounding file path- it's better bc allows for whitespaces
        name_str = "file %s" % (name) + "\n"
        print(name_str)
        L.append(name_str)

    # get file name

    # print(L)
    file.writelines(L)
    file.close()
    # copy this file to dst

    dst_path = os.path.join(
        root, name_merged_file
    )
    merged_file_path = f"{dst_path}_merged.mp4"
    if name_merged_file != "":
        cmd = f"ffmpeg -f concat -safe 0 -i mylist.txt -c copy {merged_file_path}"
        print(cmd)
        os.system(cmd)

if __name__ == "__main__":
    process_one()
