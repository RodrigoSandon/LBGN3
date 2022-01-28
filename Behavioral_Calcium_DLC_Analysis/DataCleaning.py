import os, glob
import shutil
import re


def dir_del_recursively(foldername, root) -> None:
    # all_betweencell_alignment_folders = []
    deleted = 0
    for path, dirs, files in os.walk(root):
        if foldername in dirs:
            shutil.rmtree(os.path.join(path, foldername))
            # all_betweencell_alignment_folders.append(os.path.join(path, foldername))
            deleted += 1
        print(deleted)
    print("Number of BetweenCellAlignmentData folders deleted:", deleted)


# dir_del_recursively("BetweenCellAlignmentData", ROOT)
def delete_recursively(root_path, name_endswith_list):

    for i in name_endswith_list:

        files = glob.glob(os.path.join(root_path, "**", "*%s") % (i), recursive=True)

        for f in files:
            try:
                os.remove(f)
                print("Removed ", f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))


def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            # os.remove(os.pah.join(dir,f))
            print(os.path.join(dir, f))


ROOT = r"/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData"
del_list = [
    "all_concat_cells_sorted_hm_baseline-10_-1_gauss1.5.png",
    "all_concat_cells_sorted_spaghetti_baseline-10_-1_gauss1.5.png",
]

delete_recursively(ROOT, del_list)
