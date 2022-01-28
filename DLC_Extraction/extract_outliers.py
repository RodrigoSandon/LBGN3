import deeplabcut as dlc
import os, glob


def find_paths_starts_endswith(root_path, starts, endswith) -> list:

    files = glob.glob(
        os.path.join(root_path, "**", "%s*%s") % (starts, endswith), recursive=True,
    )

    return files


def main():
    # choose from any analyzed video
    CONFIG = (
        r"/home/rory/Rodrigo/DLC_Extraction/rdt_sessions-PPT-2021-09-08/config.yaml"
    )
    ROOT = r"/media/rory/Padlock_DT/DeepLabCut_RDT_Sessions_Only"
    VIDS = find_paths_starts_endswith(ROOT, "BLA", ".avi")
    print(f"NUMBER OF VIDS: {len(VIDS)}")
    dlc.extract_outlier_frames(CONFIG, VIDS, automatic=True)


if __name__ == "__main__":
    main()
