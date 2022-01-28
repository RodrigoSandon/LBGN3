import os, glob
from shutil import copyfile


def delete_recursively(root, name_endswith_list):

    for i in name_endswith_list:

        files = glob.glob(os.path.join(root, "**", "*%s") % (i), recursive=True)

        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))


def main():
    root = r"/media/rory/Risk Videos1"
    dst = r"/media/rory/RDT VIDS/BORIS_merge"
    count = 0

    for root, dir, files in os.walk(root):
        # omit first curr dir
        if ("RRD" in root) and ("CAMERA" not in root):
            # print("FOLDER:", root)
            to_concat = False
            for name in files:
                if (
                    ("OPTO_ALL_FREE_OUTCOMES" in name)
                    or ("ALL_FREE_OUTCOMES" in name)
                    or ("OPTO_OUTCOMES" in name)
                ):
                    # print(name)
                    to_concat = True

            # so this substr does, exist, this means we follow through with this
            if to_concat == True:
                # make new dir in other master folder
                new_folder_name = os.path.join(dst, root.split("/")[4])
                # print("new folder name", new_folder_name)

                os.makedirs(new_folder_name, exist_ok=True)

                count += 1

                # open new text file
                os.chdir(root)  # in dst
                # del old mylist.txt's if any exist
                delete_recursively(root, "mylist.txt")
                delete_recursively(root, "mylist2.txt")

                print("Curr dir:", os.getcwd())
                file = open("mylist.txt", "w")
                name_merged_file = ""
                L = []
                # sort files
                files_sorted = []
                for name in files:
                    if (
                        ("OPTO_ALL_FREE_OUTCOMES" in name)
                        or ("ALL_FREE_OUTCOMES" in name)
                        or ("OPTO_OUTCOMES" in name)
                        and ("RDT" in name)
                    ):
                        files_sorted.append(name)

                files_sorted.sort()
                for name in files_sorted:
                    if (
                        ("OPTO_ALL_FREE_OUTCOMES" in name)
                        or ("ALL_FREE_OUTCOMES" in name)
                        or ("OPTO_OUTCOMES" in name)
                        and ("RDT" in name)
                    ):

                        if " " in name:
                            os.rename(
                                os.path.join(root, name),
                                os.path.join(root, name.replace(" ", "_")),
                            )
                            name = name.replace(" ", "_")
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
                        name_str = "file '%s'" % (name) + "\n"
                        print(name_str)
                        L.append(name_str)

                # get file name

                # print(L)
                file.writelines(L)
                file.close()
                # copy this file to dst
                copyfile("mylist.txt", os.path.join(new_folder_name, "mylist.txt"))
                # where to look for the individual videos
                dst_path = os.path.join(
                    new_folder_name.replace(" ", r"\ "), name_merged_file
                )
                if name_merged_file != "":
                    cmd = f"ffmpeg -f concat -safe 0 -i mylist.txt -c copy {dst_path}_merged.mp4"
                    print(cmd)
                    os.system(cmd)


if __name__ == "__main__":
    main()
