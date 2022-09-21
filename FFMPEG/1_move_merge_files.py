import shutil
import os, glob

def get_mouse_name(filename: str):
    mouse = filename.split("_")[0]

    return mouse

def find_paths_endswith(root_path, endswith: str):
    files = glob.glob(
        os.path.join(root_path, f"*{endswith}"), recursive=True,
    )
    return files


def main():
    ROOT = r"/media/rory/RDT VIDS/BORIS_merge"
    merged_files = find_paths_endswith(ROOT, "_merged.mp4")

    for file in merged_files:
        mouse = get_mouse_name(file.split("/")[-1])
        dst = os.path.join(ROOT,mouse)
        print("source:", file)
        print("destination:", dst)
        shutil.move(file, dst)

if __name__ == "__main__":
    main()