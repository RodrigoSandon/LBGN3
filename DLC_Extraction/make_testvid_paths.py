from pathlib import Path
from typing import List


def find_video_paths(raw_data_parent_dir: Path) -> List[str]:
    """Find video paths from a parent directory

    Args:
        raw_data_parent_dir (Path): Path to the parent directory
        num_vids (int): Number of video paths to return

    Returns:
        List[str]: List of string video paths
    """
    return [str(vid) for i, vid in enumerate(raw_data_parent_dir.glob("*.avi"))]


RAW_DATA_DIR = Path(r"/media/rory/RDT VIDS/DeepLabCut_RDT_Sessions_Only")
video_paths = find_video_paths(RAW_DATA_DIR)
