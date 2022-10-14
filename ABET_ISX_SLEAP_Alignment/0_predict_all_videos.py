import os, glob
from pathlib import Path

def find_paths_endswith(root_path, endswith) -> list:

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )

    return files


def main():
    ROOT = r"/media/rory/RDT VIDS/BORIS_merge/BATCH_2"
    avis = find_paths_endswith(ROOT, "merged_resized_grayscaled.mp4")
    model = "/media/rory/Padlock_DT/Opto_Speed_Analysis/models/220708_114639.centroid.n=204"
    model2 = "/media/rory/Padlock_DT/Opto_Speed_Analysis/models/220708_120742.centered_instance.n=204"

    for avi in avis:
        avi_parent = Path(avi).parents[0]
        slp_out_path = avi.replace(".mp4", ".mp4.predictions.slp")
        
        slp_file = find_paths_endswith(avi_parent, ".slp")
        
        if len(slp_file) == 0: 
            #only perform process if there arent any sleap files where merged mp4 was found
            cmd = f"sleap-track '{avi}' -m '{model}' -m '{model2}'"
            print(cmd)
            os.system(cmd)

def one_vid():
    
    video = "/media/rory/RDT VIDS/BORIS_merge/RRD43/RRD43_Risk_D1_0.1_mA_ALL_FREE_CHOICE_03282019_3_merged_resized_grayscaled.mp4"
    model = "/media/rory/Padlock_DT/Opto_Speed_Analysis/models/220708_114639.centroid.n=204"
    model2 = "/media/rory/Padlock_DT/Opto_Speed_Analysis/models/220708_120742.centered_instance.n=204"

    cmd = f"sleap-track '{video}' -m '{model}' -m '{model2}'"
    print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    #main()
    one_vid()