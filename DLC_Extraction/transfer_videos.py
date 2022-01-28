import os, glob
import shutil


def find_paths_starts_endswith(root_path, starts, endswith) -> list:

    files = glob.glob(
        os.path.join(root_path, "**", "%s*%s") % (starts, endswith), recursive=True,
    )

    return files


def main():
    ROOT = r"/media/rory/Padlock_DT/Redundant_Backup"
    DST = r"/media/rory/Padlock_DT/DeepLabCut_RDT_Sessions_Only"
    vids_to_transfer = find_paths_starts_endswith(ROOT, "BLA", ".avi")

    for i in vids_to_transfer:
        filename = i.split("/")[len(i.split("/")) - 1]
        print(f"COPYING: {filename}")
        new_path = os.path.join(DST, filename)
        print(f"NEWPATH: {new_path}")
        shutil.copy(i, new_path)


if __name__ == "__main__":
    main()
