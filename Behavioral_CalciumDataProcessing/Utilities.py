import os


def string_to_list(str, separator):
    return str.split(separator)


# example path: /media/rory/RDT VIDS/PTP_Inscopix_#1/BLA-Insc-1/Session-20210114-171459-BLA-Insc-1
def find_add_isx_session_parameters(session_path):
    found_paths = {"dff_path": "", "motion_path": "", "cellset_path": ""}
    for i in os.listdir(session_path):
        if "dff_traces.csv" in i:
            found_paths["dff_path"] = i
        elif "motion_corrected.isxd" in i:
            found_paths["motion_path"] = i
        elif "cnmfe_cellset.isxd" in i:
            found_paths["cellset_path"] = i
    return (
        found_paths["dff_path"],
        found_paths["motion_path"],
        found_paths["cellset_path"],
    )


def parse_session_id(session_path) -> str:
    lst = string_to_list(session_path, "/")
    for i in lst:
        if "Session" in i:
            return i
    return "No session id found for this path!"
