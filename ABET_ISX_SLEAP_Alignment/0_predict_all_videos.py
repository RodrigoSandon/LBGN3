import os, glob

def find_paths_endswith(root_path, endswith) -> list:

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )

    return files


def main():
    ROOT = r"/media/rory/Padlock_DT/DeepLabCut_RDT_Sessions_Only/to_be_analyzed/"
    avis = find_paths_endswith(ROOT, ".avi")
    model = "/media/rory/Padlock_DT/SLEAP/models/220204_110756.centroid.n=2835"
    model2 = "/media/rory/Padlock_DT/SLEAP/models/220204_114501.centered_instance.n=2835"

    for avi in avis:
        cmd = f"sleap-track '{avi}' -m '{model}' -m '{model2}'"
        print(cmd)
        os.system(cmd)

def one_vid():
    
    video = "/media/rory/Padlock_DT/Opto_Analysis/RRD19_Risk_0.1_mA_OPTO_ALL_FREE_OUTCOMES_01092019_2_merged.mp4"
    model = "/media/rory/Padlock_DT/SLEAP/models/220204_110756.centroid.n=2835"
    model2 = "/media/rory/Padlock_DT/SLEAP/models/220204_114501.centered_instance.n=2835"

    cmd = f"sleap-track '{video}' -m '{model}' -m '{model2}'"
    print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    #main()
    one_vid()