import pandas as pd
import os, glob
import collections, functools, operator

def find_paths(root_path: str, end: str) -> list:
    files = glob.glob(
        os.path.join(root_path, "**", end), recursive=True,
    )
    return files

def determine_mouse(cellreg_results_path):
    return cellreg_results_path.split('/')[-2]

def extract_cell_id(cell, path) -> str:
    df = pd.read_csv(path)

    return list(df[cell])[0]

def determine_cell_stability_of_mouse(cellreg_results_path, session_1, session_2, event_of_interest, cell_id_file_of_interest):

    mouse = determine_mouse(cellreg_results_path)
    print(mouse)
    df = pd.read_csv(cellreg_results_path)

    session_1_cell_lst = list(df[session_1])
    session_2_cell_lst = list(df[session_2])

    total_stability_d = {}
    total_stability_specific_d = {}

    for idx, cell_session_1 in enumerate(session_1_cell_lst):
        cell_session_2 = session_2_cell_lst[idx]
        #print(cell_session_1, cell_session_2)

        subevent_paths_cell_id_session_1 = f"{'/'.join(cellreg_results_path.split('/')[:-1])}/{session_1}/SingleCellAlignmentData/{cell_session_1}/{event_of_interest}/"
        subevent_paths_cell_id_session_2 = f"{'/'.join(cellreg_results_path.split('/')[:-1])}/{session_2}/SingleCellAlignmentData/{cell_session_2}/{event_of_interest}/"

        # each cell will have 1 event of interest but multiple subevents of interest, session_1 should be the limiting factor in terms of num of subevents
        for subevent in os.listdir(subevent_paths_cell_id_session_1):
            # each subevent will have its own these

            stability_d = {
                "stable": 0,
                "unstable": 0
            }

            stability_specific_d = {
                "+ stable": 0,
                "+ unstable": 0,
                "- stable": 0,
                "- unstable": 0,
                "N stable": 0,
                "N unstable": 0,
            }

            full_path_cell_id_session_1 = os.path.join(subevent_paths_cell_id_session_1, subevent, cell_id_file_of_interest)
            full_path_cell_id_session_2 = os.path.join(subevent_paths_cell_id_session_2, subevent, cell_id_file_of_interest)

            cell_id_session_1 = extract_cell_id(cell_session_1, full_path_cell_id_session_1)
            cell_id_session_2 = extract_cell_id(cell_session_2, full_path_cell_id_session_2)

            #print(cell_id_session_1, cell_id_session_2)

            determine_stability_of_mouse_subevent(subevent, stability_d, stability_specific_d, cell_id_session_1, cell_id_session_2)

            if subevent not in total_stability_d:
                total_stability_d[subevent] = stability_d
            else:
                total_stability_d[subevent]["stable"] += stability_d["stable"]
                total_stability_d[subevent]["unstable"] += stability_d["unstable"]

            if subevent not in total_stability_specific_d:
                total_stability_specific_d[subevent] = stability_specific_d
            else:
                total_stability_specific_d[subevent]["+ stable"] += stability_specific_d["+ stable"]
                total_stability_specific_d[subevent]["+ unstable"] += stability_specific_d["+ unstable"]
                total_stability_specific_d[subevent]["- stable"] += stability_specific_d["- stable"]
                total_stability_specific_d[subevent]["- unstable"] += stability_specific_d["- unstable"]
                total_stability_specific_d[subevent]["N stable"] += stability_specific_d["N stable"]
                total_stability_specific_d[subevent]["N unstable"] += stability_specific_d["N unstable"]

    print(total_stability_d)
    print()
    print(total_stability_specific_d)

    return total_stability_d, total_stability_specific_d

def determine_stability_of_mouse_subevent(subevent, stability_d, stability_specific_d, cell_id_session_1, cell_id_session_2):

    # doing general stability first
    if cell_id_session_1 == cell_id_session_2:
        stability_d["stable"] += 1
    elif cell_id_session_1 != cell_id_session_2:
        stability_d["unstable"] += 1
    
    # now doing specific stability
    if cell_id_session_1 == '+' and cell_id_session_1 == cell_id_session_2:
        stability_specific_d["+ stable"] += 1
    elif cell_id_session_1 == '+' and cell_id_session_1 != cell_id_session_2:
        stability_specific_d["+ unstable"] += 1
    
    if cell_id_session_1 == '-' and cell_id_session_1 == cell_id_session_2:
        stability_specific_d["- stable"] += 1
    elif cell_id_session_1 == '-' and cell_id_session_1 != cell_id_session_2:
        stability_specific_d["- unstable"] += 1
    
    if cell_id_session_1 == 'Neutral' and cell_id_session_1 == cell_id_session_2:
        stability_specific_d["N stable"] += 1
    elif cell_id_session_1 == 'Neutral' and cell_id_session_1 != cell_id_session_2:
        stability_specific_d["N unstable"] += 1



def main():
    # example
    # path to cellreg results for mouse: /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/cellreg_Pre-RDT RM_RDT D1.csv

    # session 1 cell id path: /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/Pre-RDT RM/SingleCellAlignmentData/C01/Block_Reward Size_Shock Ocurred_Choice Time (s)/(1.0, 'Large', False)/id_z_fullwindow_auc_bonf0.05_71_101_101_131.csv

    # session 2 cell id path: /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/SingleCellAlignmentData/C01/Block_Reward Size_Shock Ocurred_Choice Time (s)/(1.0, 'Large', False)/id_z_fullwindow_auc_bonf0.05_71_101_101_131.csv
    session_types = [
            
            "Pre-RDT RM",
            "RDT D1",
    ]

    event_of_interest = "Block_Reward Size_Shock Ocurred_Choice Time (s)"

    cell_id_file_of_interest = "id_z_fullwindow_auc_bonf0.05_71_101_101_131.csv"

    mice = [
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-2",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-3",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-5",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-7",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-8",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-9",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-11",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-13",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-14",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-15",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-16",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-18",
        "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-19",
        ]

    mice_cellreg_paths = ["/".join([i,"cellreg_Pre-RDT RM_RDT D1.csv"]) for i in mice]

    # Based on the cellreg cell registration, determine percentage stability of each cell
    overall_stability_lst_of_dicts = []
    overall_stability_specific_lst_if_dicts = []

    for mouse_cellreg_path in mice_cellreg_paths:
        try:
            total_stability_d, total_stability_specific_d = determine_cell_stability_of_mouse(mouse_cellreg_path, session_types[0], session_types[1], event_of_interest, cell_id_file_of_interest)
            overall_stability_lst_of_dicts.append(total_stability_d)
            overall_stability_specific_lst_if_dicts.append(total_stability_specific_d)

        except FileNotFoundError as e:
            print(e)
            pass

    result_1 = {}
    sum_dicts(result_1, overall_stability_lst_of_dicts)
            
    print("total_stability_d across mice : ", str(result_1))

    result_2 = {}
    sum_dicts(result_2, overall_stability_specific_lst_if_dicts)
            
    print("total_stability_specific_d across mice : ", str(result_2))

def sum_dicts(result_d, list_of_ds):
    for d in list_of_ds:
        for k in d.keys():
            if k not in result_d:
                result_d[k] = {}
            for k2 in d[k].keys():
                if k2 not in result_d[k]:
                    result_d[k][k2] =  d[k][k2]
                else:
                    result_d[k][k2] += d[k][k2]


if __name__ == "__main__":
    main()