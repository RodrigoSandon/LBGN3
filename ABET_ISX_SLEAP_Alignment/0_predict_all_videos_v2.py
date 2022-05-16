import os, glob
from pathlib import Path

def find_paths_endswith(root_path, endswith) -> list:

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )

    return files


def main():
    ROOT = r"/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/"
    avis = find_paths_endswith(ROOT, ".avi")
    model = "/media/rory/Padlock_DT/SLEAP/models/220204_110756.centroid.n=2835"
    model2 = "/media/rory/Padlock_DT/SLEAP/models/220204_114501.centered_instance.n=2835"

    for avi in avis:
        slp = f"{avi}.predictions.slp"
        slp = Path(slp)
        if slp.is_file():
            pass
        else: # start where you left off
            cmd = f"sleap-track '{avi}' -m '{model}' -m '{model2}'"
            print(cmd)
            os.system(cmd)

if __name__ == "__main__":
    main()