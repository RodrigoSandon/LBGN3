"""
Goals are the following:
    1) To extract h5 files from slp files
    2) Automate this conversion for 219 .slp files
    3) Extract this h5 file to its respective folder.

"""

import os, glob
import shutil

def find_paths_endswith(root_path, endswith) -> list:

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )

    return files


def slp_to_h5(in_path, out_path):

    cmd = f"sleap-convert --format analysis -o {out_path} {in_path}"
    os.system(cmd)

def slp_file_parsing(slp_filename: str):
    slp_filename = slp_filename.split("/")[-1]
    mouse = slp_filename.split("_")[0]
    session = "_".join(slp_filename.split("_")[1:]).replace(".avi.predictions.slp","")

    try:
        session_mod_1 = session.split("20")[0]

        if "." in session_mod_1:
            session_mod_1 = session_mod_1.replace(".","")
        if "_" in session_mod_1:
            session_mod_1 = session_mod_1.replace("_", " ")

        #print(f"{mouse}: {session_mod_1}")

        return mouse, session_mod_1
    except Exception as e:
        print(f"Session {mouse}: {session} can't be renamed!")
        print(e)
        pass

# e1: /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-8/RDT D1/Session-20211021-093007_BLA-INSC-8-RDT-D1
# e2: /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1 <- (bla1-3)
# structure1: /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#{}/BLA-Insc-{}/{}
# structure2: /media/rory/Padlock_DT/BLA_Analysis/{}/BLA-Insc-{}/{}/{}
#                                     {if blax-x -> pertain to this ptp}    {y}{y}{startswith "session"}/->GO

def find_dst_for_mouse_and_session(base, mouse_tolower: str, session_tolower: str) -> str:
    # base = /media/rory/Padlock_DT/BLA_Analysis/

    # 1) Find ptp folder
    def ptp_autoencoder(mouse_tolower) -> str:
        """Gives you a ptp #, based on mouse name."""
        d = {
            "PTP_Inscopix_#1" : ["bla-insc-1","bla-insc-2","bla-insc-3"],
            "PTP_Inscopix_#3" : ["bla-insc-5","bla-insc-6","bla-insc-7"],
            "PTP_Inscopix_#4" : ["bla-insc-8","bla-insc-9","bla-insc-11","bla-insc-13"]
        }
        
        for key in d.keys():
            if mouse_tolower in d[key]:
                return key

    # 2) Find the appropriate mouse #, since capitalization is diff in the dst dir
    def mouse_name_autoencoder(mouse_tolower) -> int:
        """Gives you a number based on the number that's in the input."""
        num = int(mouse_tolower.split("-")[-1])
        #print(num)
        return num

    # 3) Session str is already properly formatted, nothing to do here.
    # 4) Need to find the interfolder (only exists for bla5-13)
    def add_interfolder(curr_dir) -> str:
        # So far have this: /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-8/RDT D1/

        for dir in os.listdir(curr_dir):
            # if the substrings that identified a interfolder are present, then add that to dst
            #print(dir)
            if "-" in dir:
                return dir
            else: #weird error, not getting added for one
                return "Session-20211021-093007_BLA-INSC-8-RDT-D1"
    
    ptp_folder = ptp_autoencoder(mouse_tolower)

    # Add ptp folder specification
    dst = os.path.join(base, ptp_folder)
    # Add mouse folder specification
    dst = os.path.join(dst,f"BLA-Insc-{mouse_name_autoencoder(mouse_tolower)}")
    # Add session type specification
    dst = os.path.join(dst, session_tolower.upper())
    isdir = os.path.isdir(dst)

    if not isdir:
        dst = dst + " NEW_SCOPE"

    # ^ this is full path for bla1-3

    # Because only the other PTP folders have interfolders
    if ptp_folder != "PTP_Inscopix_#1":
        dst = os.path.join(dst, add_interfolder(dst))

    #print(dst)
    return dst

def send_bla_files_somewhere(base, bla_slp_files: list):
    for i in bla_slp_files:
        mouse, session = slp_file_parsing(i) 
        mouse = mouse.lower()
        session = session.lower()
        dst = find_dst_for_mouse_and_session(base, mouse, session)
        print(f"SLP destination for {mouse} {session}: {dst}")


def send_all_other_files_somewhere(other_slp_files: list):
    for i in other_slp_files:
        slp_file_parsing(i)

def main():
    ROOT = r"/media/rory/Padlock_DT/DeepLabCut_RDT_Sessions_Only"
    BLA_DST_ROOT = r"/media/rory/Padlock_DT/BLA_Analysis"
    OTHER_DST_ROOT = r"/media/rory/Padlock_DT/Non-BLA_Analysis"

    slp_files = find_paths_endswith(ROOT, ".slp")

    bla_slp_files = [i for i in slp_files if "BLA" in i]
    other_slp_files = [i for i in slp_files if "BLA" not in i]

    print("===== PROCESSING BLA FILES =====")
    print(f"Number of BLA SLP files: {len(bla_slp_files)}")

    """Will be actually putting the bla's into existing folders"""
    send_bla_files_somewhere(BLA_DST_ROOT, bla_slp_files)

    """Other folders will go into a new root folder"""
    print("===== PROCESSING OTHER FILES =====")
    send_all_other_files_somewhere(other_slp_files)





if __name__ == "__main__":
    main()