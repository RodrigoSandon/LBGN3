import os, glob

def find_paths_endswith(root_path, endswith) -> list:

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )

    return files


def main():
    ROOT = r"/media/rory/Padlock_DT/DeepLabCut_RDT_Sessions_Only"
    avis = find_paths_endswith(ROOT, ".avi")
    model = "/media/rory/Padlock_DT/SLEAP/models/220204_110756.centroid.n=2835"
    model2 = "/media/rory/Padlock_DT/SLEAP/models/220204_114501.centered_instance.n=2835"

    for avi in avis:
        cmd = f"sleap-track '{avi}' -m '{model}' -m '{model2}'"
        print(cmd)
        os.system(cmd)

if __name__ == "__main__":
    main()