import sys
import os
import subprocess

def get_frame_rate(filename):
    if not os.path.exists(filename):
        sys.stderr.write("ERROR: filename %r was not found!" % (filename,))
        return -1         
    out = subprocess.check_output(["ffprobe",filename,"-v","0","-select_streams","v","-print_format","flat","-show_entries","stream=avg_frame_rate"])
    rate = str(out).split('=')[1].replace("\"", "").replace("\'","").replace("\\n", "").split('/')
    #print(rate)
    if len(rate)==1:
        return float(rate[0])
    if len(rate)==2:
        return float(rate[0])/float(rate[1])
    return -1

def main():
    vid = "/media/rory/RDT VIDS/BORIS_merge/RRD36/RRD36_Risk_D1_0.1_mA_ALL_FREE_CHOICE_03282019_3_trimed_merged_resized_grayscaled.mp4"
    fps = get_frame_rate(vid)

    print(fps)

if __name__ == "__main__":
    main()