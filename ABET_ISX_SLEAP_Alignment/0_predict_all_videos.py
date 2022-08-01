import os, glob

def find_paths_endswith(root_path, endswith) -> list:

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )

    return files


def main():
    ROOT = r"/media/rory/RDT VIDS/BORIS"
    avis = find_paths_endswith(ROOT, "merged_resized_grayscaled.mp4")
    model = "/media/rory/Padlock_DT/Opto_Analysis/models/220708_114639.centroid.n=204"
    model2 = "/media/rory/Padlock_DT/Opto_Analysis/models/220708_120742.centered_instance.n=204"

    for avi in avis:
        slp_out_path = avi.replace(".mp4", ".mp4.predictions.slp")
        
        if os.path.isfile(slp_out_path) == False:
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
    main()
    #one_vid()