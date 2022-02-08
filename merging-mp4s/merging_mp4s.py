import os
import glob
from pathlib import Path


def main():
    count = 0
    ROOT_PATH = Path(r"/media/rory/RDT VIDS/BORIS_merge")

    for root, dir, files in os.walk(ROOT_PATH):
        # omit first curr dir
        if root != "/media/rory/RDT VIDS/BORIS_merge":
            # change working directory
            os.chdir(root)
            print("Current working dir: ", os.getcwd())
            count += 1

            # open new text file
            file = open("mylist.txt", "w")
            name_merged_file = ""
            L = []
            for name in files:
                # print(name)
                if "mylist.txt" or "_merged.mp4" not in name:
                    if ".MP4" in name:
                        if " " in name:
                            os.rename(name, name.replace(" ", "_"))
                            name = name.replace(" ", "_")
                        if "(" and ")" in name:
                            os.rename(name, name.replace("(", "").replace(")", ""))
                            name = name.replace("(", "").replace(")", "")
                        # change acc folder name
                        # and change variable name accordingly (change it the same way)
                        # and make sure this is reflected in mylist.txt
                    name_merged_file = name.replace(".MP4", "")
                    name_str = "file '%s'" % (name) + "\n"
                    print(name_str)
                    L.append(name_str)
            print(L)
            # get file name

            # print(L)
            file.writelines(L)
            file.close()
            # have the input file written now
            if name_merged_file != "":
                cmd = f"ffmpeg -f concat -safe 0 -i mylist.txt -c copy {name_merged_file}_merged.mp4"
                print(cmd)
                os.system(cmd)


if __name__ == "__main__":
    ROOT = Path(r"/media/rory/RDT VIDS/BORIS_merge")
    main()
    #input_y_n = input("Would you like to delete the old mp4s (y/n)?")

    """if input_y_n == "y":
        delete_old_mp4s(ROOT)
    elif input_y_n == "n":
        pass
"""