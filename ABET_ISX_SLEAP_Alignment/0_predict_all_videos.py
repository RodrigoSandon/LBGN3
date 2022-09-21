import os, glob

def find_paths_endswith(root_path, endswith) -> list:

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )

    return files


def main():
    ROOT = r"/media/rory/RDT VIDS/BORIS_merge"
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
    
    video = "/media/rory/RDT VIDS/BORIS/RRD124_RDT_OPTO_CHOICE_03112020_5_merged_resized_grayscaled.mp4"
    model = "/media/rory/Padlock_DT/Opto_Analysis/models/220708_114639.centroid.n=204"
    model2 = "/media/rory/Padlock_DT/Opto_Analysis/models/220708_120742.centered_instance.n=204"

    cmd = f"sleap-track '{video}' -m '{model}' -m '{model2}'"
    print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    main()
    #one_vid()