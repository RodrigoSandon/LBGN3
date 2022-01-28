from pathlib import Path
import os
from typing import List
from preprocess import preprocess

ROOT_DIR = Path(r"/media/rory/PTP Inscopix 1/Inscopix/Raw Inscopix Data Files")


def find_video_paths(root_directory: Path) -> List[Path]:

    vids = []
    numberISXDFiles = 0

    for root, dirs, files in os.walk(root_directory):
        print(root)
        if "Good Sessions" in root:
            for name in files:
                if name.endswith(".isxd"):
                    numberISXDFiles += 1
                    file = os.path.join(root, name)
                    vids.append(file)
    print("number of files: ", numberISXDFiles)
    print(*vids, sep="\n")
    vids.reverse()  # reverse just to try to get a less huge file on first iter
    return vids


def main() -> None:
    vids = find_video_paths(ROOT_DIR)
    print(vids)


if __name__ == "__main__":
    main()
