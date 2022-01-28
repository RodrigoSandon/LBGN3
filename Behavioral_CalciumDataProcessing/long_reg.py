import isx
import os
import glob
from pathlib import Path
from typing import List

root_path = r"/media/rory/RDT VIDS/PTP_Inscopix_#3/BLA-Insc-6"

# example: /media/rory/RDT VIDS/PTP_Inscopix_#3/BLA-Insc-6/Session-20210503-121350-BLA-Insc-6_BLA-Insc-6_RM_D1/2021-05-03-12-23-57_video_BLA-Insc-6_RM_D1/cell__C47.
def delete_recursively(root_dir, name_endswith_list):
    for i in name_endswith_list:

        files = glob.glob(
            os.path.join(root_dir, "**", "**", "*%s") % (i), recursive=True
        )
        print(files)

        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))


def vids_to_process(dir: Path) -> List[Path]:

    vids = []
    movies = []

    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.startswith("cnmfe_cellset.isxd"):
                vids.append(os.path.join(root, name))

    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.startswith("motion_corrected.isxd"):
                movies.append(os.path.join(root, name))

    vids_out = []
    movies_out = []
    # ex: /media/rory/RDT VIDS/PTP_Inscopix_#3/BLA-Insc-6/Session-20210503-121350-BLA-Insc-6_BLA-Insc-6_RM_D1/2021-05-03-12-23-57_video_BLA-Insc-6_RM_D1/cnmfe_cellset.isxd
    for i in range(len(vids)):
        mod = vids[i].replace("cnmfe_cellset.isxd", "cnmfe_cellsets_out")

        root1 = vids[i].replace("/cnmfe_cellset.isxd", "")
        mod_1 = isx.make_output_file_path(mod, root1, suffix=None, ext="isxd")
        #  with open(mod_1, "w") as fp:
        #    pass
        print(mod_1)
        vids_out.append(mod_1)

    return vids, vids_out


def main():
    result_csv_out_path = (
        r"/media/rory/RDT VIDS/PTP_Inscopix_#3/BLA-Insc-6/longreg_results.csv"
    )

    list_vids, list_vids_output = vids_to_process(root_path)
    print(list_vids_output)
    isx.longitudinal_registration(
        list_vids,
        list_vids_output,
        csv_file=result_csv_out_path,
        min_correlation=0.95,
        accepted_cells_only=True,
    )


def main2():
    """Give list of substrings that file names end with"""

    to_delete = ["cellsets_out.isxd", ".", "_traces.csv"]
    delete_recursively(root_path, to_delete)


if __name__ == "__main__":
    main2()
