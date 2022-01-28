from typing import List
from deeplabcut import create_new_project
from pathlib import Path


def find_video_paths(raw_data_parent_dir: Path, num_vids: int) -> List[str]:
    """Find video paths from a parent directory

    Args:
        raw_data_parent_dir (Path): Path to the parent directory
        num_vids (int): Number of video paths to return

    Returns:
        List[str]: List of string video paths
    """
    return [
        str(vid)
        for i, vid in enumerate(raw_data_parent_dir.glob("*.avi"))
        if i < num_vids
    ]


RAW_DATA_DIR = Path(r"/media/rory/RDT VIDS/DeepLabCut_RDT_Sessions_Only")
NUM_VIDS = 40


def main():
    project_path = Path(".").absolute().parent
    video_paths = find_video_paths(RAW_DATA_DIR, num_vids=NUM_VIDS)
    create_new_project(
        "rdt_sessions",
        experimenter="PPT",
        working_directory=str(project_path),
        videos=video_paths,
    )


if __name__ == "__main__":
    main()
