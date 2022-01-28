from pathlib import Path
from typing import List, Optional, Dict
import isx


def preprocess(in_path: Path, out_dir: Optional[Path] = None):
    """Preprocess a single insopix video

    Args:
        in_path (Path): Path to inscopix .isxd file
        out_dir (Optional[Path], optional): Path where files will be saved. Defaults to directory of input video.
    """

    def _generate_default_outpath(inpath: Path) -> Path:
        return inpath.parent

    def _create_paths(in_path: Path, out_path: Path) -> Dict[str, Path]:
        out = {}
        out["downsample_path"] = out_path / f"downsampled_gSiz{str(gSiz)}.isxd"
        out["spatial_filter_path"] = out_path / f"spatial_filtered_gSiz{str(gSiz)}.isxd"
        out["motion_correct_path"] = out_path / f"motion_corrected_gSiz{str(gSiz)}.isxd"
        out["cnmfe_path"] = out_path / f"cnmfe_cellset_gSiz{str(gSiz)}.isxd"
        for path in out.values():
            if path.exists():
                path.unlink()
        return out

    def _downsample(
        in_vid: Path,
        out_vid: Path,
        temporal_factor: float = 2,
        spatial_factor: float = 4,
    ):
        print("downsampling...")
        isx.preprocess(
            str(in_vid),
            str(out_vid),
            temporal_downsample_factor=temporal_factor,
            spatial_downsample_factor=spatial_factor,
        )

    def _spatial_filter(
        in_vid: Path, out_vid: Path, low_cutoff: float = 0.005, high_cutoff: float = 0.5
    ):
        print("applying spatial filter...")
        isx.spatial_filter(
            str(in_vid), str(out_vid), low_cutoff=low_cutoff, high_cutoff=high_cutoff
        )

    def _motion_correct(
        in_vid: Path,
        out_vid: Path,
        max_translation: int = 20,
        low_bandpass_cutoff: float = 0.054,
        high_bandpass_cutoff=0.067,
    ):
        print("motion correcting...")
        out_motion = out_vid.parent / "motion_ts.csv"
        isx.motion_correct(
            str(in_vid),
            str(out_vid),
            max_translation=max_translation,
            low_bandpass_cutoff=low_bandpass_cutoff,
            high_bandpass_cutoff=high_bandpass_cutoff,
            output_translation_files=str(out_motion),
        )

    def _cnmfe(
        in_vid: Path,
        out_dir: Path,
        num_threads,
        gSiz,
        min_corr,
        min_pnr,
        bg_spatial_subsampling,
        ring_size_factor,
        gSig,
        closing_kernel_size,
        merge_threshold,
        processing_mode,
        patch_size,
        patch_overlap,
        output_unit_type,
    ):
        print("running cnmfe")
        isx.run_cnmfe(
            input_movie_files=[str(in_vid)],
            output_cell_set_files=[str(out_dir)],
            output_dir=str(out_dir.parent),
            num_threads=num_threads,
            cell_diameter=gSiz,
            min_corr=min_corr,
            min_pnr=min_pnr,
            bg_spatial_subsampling=bg_spatial_subsampling,
            ring_size_factor=ring_size_factor,
            gaussian_kernel_size=gSig,
            closing_kernel_size=closing_kernel_size,
            merge_threshold=merge_threshold,
            processing_mode=processing_mode,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            output_unit_type=output_unit_type,
        )

    if out_dir is None:
        out_dir = _generate_default_outpath(in_path)

    # CHANGE PARAMETERS HERE
    # slightlty better in catching missed cells, but leaves some out, still slightly less fewer misses as a net result
    # left a big cell out and one cell which pnr was low, so we lowered pnr
    #  update 10/18/21 - need to be more lineant as to what is considered a cell
    num_threads = 5  # staying the same
    gSiz = 8  # cell diameter needs o account for bigger cells, was 16 10/19/21, now 48, then 8
    min_corr = 0.7
    min_pnr = 8  # before was 8 10/18/21, 5 on 10/19/21, changed it back to 8 11/22/21
    bg_spatial_subsampling = 1
    ring_size_factor = 1.125
    gSig = 4  # 4 10/18/21
    closing_kernel_size = 0
    merge_threshold = 0.3
    processing_mode = "parallel_patches"
    patch_size = 80
    patch_overlap = 20
    output_unit_type = "df_over_noise"

    out_paths = _create_paths(in_path=in_path, out_path=out_dir)
    _downsample(in_path, out_paths[f"downsample_path"])
    _spatial_filter(out_paths[f"downsample_path"], out_paths[f"spatial_filter_path"])
    _motion_correct(
        out_paths[f"spatial_filter_path"], out_paths[f"motion_correct_path"]
    )
    _cnmfe(
        out_paths[f"motion_correct_path"],
        out_paths[f"cnmfe_path"],
        num_threads,
        gSiz,
        min_corr,
        min_pnr,
        bg_spatial_subsampling,
        ring_size_factor,
        gSig,
        closing_kernel_size,
        merge_threshold,
        processing_mode,
        patch_size,
        patch_overlap,
        output_unit_type,
    )
