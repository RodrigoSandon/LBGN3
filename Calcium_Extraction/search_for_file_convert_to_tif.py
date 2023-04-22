import isx 
import os, glob


def find_paths(root_path, endswith: str):
    files = glob.glob(
        os.path.join(root_path, "**", endswith), recursive=True,
    )
    return files

def main():

    ROOT = r"/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1" # change ur root path here

    # find your file paths containing the motion_corrected.isxd ending
    
    my_list = find_paths(ROOT, "motion_corrected.isxd")
    print(my_list)

    for file_path in my_list:
        replacement_path = file_path.replace(".isxd", ".tif")
        isx.export_movie_to_tiff(file_path, replacement_path)

if __name__ == "__main__":
    main()