from pathlib import Path
from typing import List
import glob, os, isx
import pandas as pd
import numpy as np

"""
Summary:

    For one mouse:

    1) Export dff traces for each session under a given root file.
    2) Preprocess dff traces and export the result to its corresponding folder.
    3) In parallel, perform and export the longitudinal registration 
    results for all the sessions that contain cnmfe_cellset.isxd files.
    4) Preprocess longreg file and export result.
    5) make correct abet gpio file
"""
"""
Methods:

    export_session_dff

    Returns:
        CSV FILE: returns a csv file to located folders containing all the df/f traces 
        for all the cells identified for that session as columns, and the rows represent 
        time points at which all these df/f traces were recorded at simultaneously.

    long_reg

    Returns:
        CSV FILE: containing he meta data of the longreg results
        ISXD FILES: an output cnmfe_cellset isxd file is outputted to 
        same directory where its corresponding original cnmfe_cellset 
        file was found in.

    get_input_cellset_files

    Returns:
        LIST OF CELL SET FILE PATHS: returns a list of cell set file paths that will 
        be needed to iterate over so we can call export_session_df individually on 
        those paths.

        LIST OF CELL SET FILE ROOT PATHS: returns a list of the roots of these cell set 
        files paths so that we are able to locate the outputs of the export_session_df to 
        the same folder the cell set was in.

    delete_recursively

    Returns:
        VOID: deletes files that contain any substring in the input 
        list within a given root directory.

    premake_listof_outfile_paths

    Returns:
        LIST OF OUT FILE PATHS: long. reg. take a list of things for input and output 
        and can't do individual, so therefore need premade out files

    preprocess_longreg_results

    Returns:
        CSV FILE: 
            1) filtering out rows by ncc score
            2) replacing local_cell_index with cell's actual name
            3) replace local_cellset_index 

    preprocess_dff_traces_csv

    Returns:
        CSV FILE:
            1) Filter out columns by cells who were rejected
            2) Remove the "accepted" part o the row and replace it with
            the actual cell names
            3) Delete the first row (should be all blank now)
            4) Change the cell names to go from 00 to XX
    
    find_recursively

    Returns:
        LIST OF CSV PATHS: Based on how a file ends, get a list of csv files that are
        in the directory, recursively.

"""


class CalciumPreprocessing:
    def __init__(self, root_path, raw_root_path):
        self.root_path = root_path
        self.raw_root_path = raw_root_path

    def export_session_dff(
        self,
        input_cell_set_files,
        output_csv_file,
        output_tiff_file,
        time_ref,
        output_props_file,
    ):
        isx.export_cell_set_to_csv_tiff(
            input_cell_set_files,
            output_csv_file,
            output_tiff_file,
            time_ref,
            output_props_file,
        )

    def get_input_cell_set_files(self):
        cell_set_files = []
        root_paths_to_cell_set_files = []

        for root, dirs, files in os.walk(self.root_path):
            for name in files:

                if name.startswith("cnmfe_cellset.isxd"):
                    path_to_cellset = os.path.join(root, name)
                    cell_set_files.append(path_to_cellset)
                    root_paths_to_cell_set_files.append(root)

        print("Number of cell sets: %s" % (len(cell_set_files)))
        return cell_set_files, root_paths_to_cell_set_files

    def delete_recursively(self, name_endswith_list):

        for i in name_endswith_list:

            files = glob.glob(
                os.path.join(self.root_path, "**", "*%s") % (i), recursive=True
            )

            for f in files:
                try:
                    os.remove(f)
                except OSError as e:
                    print("Error: %s : %s" % (f, e.strerror))

    def find_recursively(self, filename_endswith):

        files = glob.glob(
            os.path.join(self.root_path, "**", "*%s") % (filename_endswith),
            recursive=True,
        )

        return files

    def find_recursively_newroot(self, root_path, filename_endswith):

        files = glob.glob(
            os.path.join(root_path, "*%s") % (filename_endswith),
            recursive=True,
        )

        return files

    def find_raw_recursively(self, filename_endswith):

        files = glob.glob(
            os.path.join(self.raw_root_path, "**", "*%s") % (filename_endswith),
            recursive=True,
        )

        return files

    def premake_listof_outfile_paths(self, root_paths: List, out_name):
        return [os.path.join(i, out_name) for i in root_paths]

    def preprocess_longreg_results(
        self, longreg_file, ncc_score_threshold, root_list_of_cnmfe_cellsets
    ):
        def filter_glob_cells_by_ncc_score(df, ncc_score_threshold):
            return df[df["ncc_score"] >= ncc_score_threshold]

        def idx_to_name(x):
            if len(str(x)) == 1:
                return "C0" + str(x)
            else:
                return "C" + str(x)

        def modify_local_cell_index_col(df):
            # first rename the column
            df = df.rename(columns={"local_cell_index": "local_cell_name"})
            # then change the values of all the rows
            df["local_cell_name"] = df["local_cell_name"].apply(idx_to_name)

            return df

        # ex: /media/rory/PTP Inscopix 2/PTP_Inscopix_#3/BLA-Insc-6/Session-20210518-102215_BLA-Insc-6_RDT_D1/2021-05-18-10-26-03_video_BLA-Insc-6_RDT_D1/cnmfe_cellset.isxd
        # ex 12/22/21: /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#3/BLA-Insc-6/RM D9/2021-05-14-10-14-17_video_BLA-Insc-6_RM_D9/cnmfe_cellset.isxd
        def modify_local_cellset_index_col(df, root_list_of_cnmfe_cellsets):
            def get_session_name(
                session_idx_in_mouse_folder, root_list_of_cnmfe_cellsets
            ):
                # print(root_list_of_cnmfe_cellsets) good
                # print("Session idx: ", session_idx_in_mouse_folder)
                for i in range(len(root_list_of_cnmfe_cellsets)):
                    # print(i) good
                    if i == session_idx_in_mouse_folder:  # if the indices are the same
                        # print(root_list_of_cnmfe_cellsets[i].split("/")[6])
                        return root_list_of_cnmfe_cellsets[i].split("/")[7]
                        # Changed where to find session name for long reg results 12/22/21
                        # get that session name based on idx

            # change the values of all the rows
            new_col_vals = []
            for row_idx in range(len(df)):
                session_idx_in_mouse_folder = df.iloc[row_idx]["local_cellset_index"]
                # print(session_idx_in_mouse_folder)
                session_name = get_session_name(
                    session_idx_in_mouse_folder, root_list_of_cnmfe_cellsets
                )
                # df.loc[
                #   row_idx, df.columns.get_loc("local_cellset_index")
                # ] = session_name  # using ".at[]" needs all indicies
                new_col_vals.append(session_name)

            # rename the column

            df = df.rename(columns={"local_cellset_index": "session_name"})
            df["session_name"] = new_col_vals

            return df

        df = pd.read_csv(longreg_file)
        df = filter_glob_cells_by_ncc_score(df, ncc_score_threshold)
        df = modify_local_cell_index_col(df)
        df = modify_local_cellset_index_col(df, root_list_of_cnmfe_cellsets)

        return df

    def preprocess_dff_traces_csv(self, csv_path):
        # remove rejected columns
        df = pd.read_csv(csv_path)
        # these tables have hidden whitespaces in the names/values
        df = df.loc[:, ~(df == " rejected").any()]
        # df = df.T
        # print(df.columns)
        # print(list(df.columns.values.tolist())[1])
        # df = df[df[df.columns[1]].str.contains(" rejected") == False]
        # df = df.T
        # identifying name and replacing col name
        to_rename_colname = df.columns[0]
        # print("'", to_rename_colname, "'", sep="")
        df = df.rename(columns={" ": "Time(s)"})
        # print(df.head())

        # dropping 1st row
        df = df.drop([0])
        # print(df.head())

        def idx_to_name(x):
            if len(str(x)) == 1:
                return "C0" + str(x)
            else:
                return "C" + str(x)

        # renaming columns (no skipping of the number of the cell)
        for idx, colname in enumerate(list(df.columns)):
            if colname != "Time(s)":
                new_name = idx_to_name(idx)
                df = df.rename(columns={colname: new_name})

        # print(df)
        return df

    def export_gpio(
        self, gpio_path, root_to_out
    ) -> Path:  # returns path where csv is saved
        if gpio_path == "":
            print("GPIO file not found!")
        else:  # want output csv file to be where it's processed file is
            isx.export_gpio_set_to_csv(
                gpio_path,
                str(root_to_out + "/gpio.csv"),
                str(root_to_out),
                time_ref="start",
            )
        return root_to_out + "/gpio.csv"

    def get_time_gpio_starts(
        self, gpio_df, stop_at=2
    ):  # at the second time you see "GPIO-1", that's when you choose the "Time (s)" value
        count = 0
        time_gpio_starts = 0
        # print(gpio_csv.head())
        # print(list(gpio_csv.columns))
        for row_idx in range(len(gpio_df)):
            if "GPIO-1" in gpio_df.loc[row_idx, " Channel Name"]:
                count += 1
                if count == stop_at:
                    time_gpio_starts = gpio_df.loc[row_idx, "Time (s)"]
                    return time_gpio_starts
        return "No start time found!"

    def delete_certain_rows_of_df(
        self, dff_traces, remove_rows_with_value_below_this_number
    ) -> pd.core.frame.DataFrame:
        dff_traces = dff_traces.apply(pd.to_numeric)
        dff_traces = dff_traces[
            dff_traces.loc[:, "Time(s)"] >= remove_rows_with_value_below_this_number
        ]
        return dff_traces

    def is_shock_session(self, abet_path) -> bool:
        if "Shock Test" in abet_path:
            return True
        else:
            return False

    def add_gpio_start_time_to_ABET_cols(self, abet_path, gpio_start_time):
        df_abet = pd.read_csv(abet_path)
        # print(df_abet.head())
        """print(
            type(df_abet["Start Time (s)"]),
            df_abet["Start Time (s)"],
            type(gpio_start_time),
            gpio_start_time,
        )"""
        if self.is_shock_session(abet_path) == True:
            df_abet["Shock Time (s)"] = df_abet["Shock Time (s)"] + gpio_start_time

        elif self.is_shock_session(abet_path) == False:
            df_abet["Start Time (s)"] = df_abet["Start Time (s)"] + gpio_start_time
            df_abet["Choice Time (s)"] = df_abet["Choice Time (s)"] + gpio_start_time
            df_abet["Collection Time (s)"] = (
                df_abet["Collection Time (s)"] + gpio_start_time
            )
        # print(df_abet.head())
        return df_abet


def main():
    """
    Notes:
        - Length of "cellsets" and "roots" should have the same length, therefore the same index.
        - We use the same list of file paths that are in "cellsets" to perform long.reg.
        - We use the same list of root paths that are in "roots" for the individual part of
        the long. reg. net output (net long. reg. output = single csv file telling us global
        cell idx and ncc values + indv. cellset output isxd files).
    """
    """TODO: 
    1) overwrite function, overwrite existing files or no? 
    -saves time if just adding on things curr folder
    """

    # Needs to be left like this because where the raw files are can change
    MASTER_PATH = r"/media/rory/Padlock_DT/BLA_Analysis"
    MASTER_RAW_PATH = r"/media/rory/Nathen's Fantom/Inscopix_Raw_Data_Organized"  # Change ideally, so that you won't have to repeat calcium preprocessing
    MICE_PATHS_IN_RAW = [
        os.path.join(MASTER_RAW_PATH, mouse) for mouse in os.listdir(MASTER_RAW_PATH)
    ]

    for folder in os.listdir(MASTER_PATH):
        if "PTP" in folder:
            BATCH_PATH = os.path.join(MASTER_PATH, folder)
            for mouse in os.listdir(BATCH_PATH):
                if "BLA" in mouse:
                    MOUSE_ROOT_PATH = os.path.join(BATCH_PATH, mouse)
                    RAW_ROOT_PATH = None
                    for raw_mice_path in MICE_PATHS_IN_RAW:
                        if mouse == raw_mice_path.split("/")[-1]:
                            RAW_ROOT_PATH = raw_mice_path

                    print(f"Mouse processed path: {MOUSE_ROOT_PATH}")
                    print(f"Mouse raw path: {RAW_ROOT_PATH}")
                    # Indicate for what root directory are these utilities for
                    util = CalciumPreprocessing(MOUSE_ROOT_PATH, RAW_ROOT_PATH)

                    cellsets, roots = util.get_input_cell_set_files()
                    # example: /media/rory/PTP Inscopix 2/PTP_Inscopix_#3/BLA-Insc-5/Session-20210510-093930_BLA-Insc-5_RM_D1/2021-05-10-09-45-37_video_BLA-Insc-5_RM_D1/cnmfe_cellset.isxd
                    # session name is at the [6] split
                    # Identifies all cellsets in mouse and its according root paths for cellsets

                    # Make sure output files don't already exist- else an error
                    to_delete = [
                        ".",
                        "dff_traces.csv",
                        "gpio.csv",
                        "dff_traces_preprocessed_gpio.csv",
                    ]
                    print("Deleting files that contain the string(s): %s" % (to_delete))
                    util.delete_recursively(to_delete)

                    # Exporting csv dff traces to their corresponding session directories.
                    for i in range(len(cellsets)):
                        util.export_session_dff(
                            cellsets[i],
                            os.path.join(roots[i], "dff_traces.csv"),
                            os.path.join(roots[i], "cell_"),
                            "start",
                            "",
                        )

                    # We don't need the tiff files that come with "export_session_dff()", so delete them
                    to_delete = ["."]
                    print("Deleting files that contain the string(s): %s" % (to_delete))
                    util.delete_recursively(to_delete)

                    # Perform long. reg. on the, but first, make cellset output files

                    # do this deletion of out_files to make sure an error that it already exists does not appear
                    to_delete = ["cnmfe_cellset_out.isxd", "longreg_results.csv"]
                    print("Deleting files that contain the string(s): %s" % (to_delete))
                    util.delete_recursively(to_delete)

                    meta_csv_filename = os.path.join(
                        util.root_path, "longreg_results.csv"
                    )

                    print(
                        len(cellsets),
                        len(
                            util.premake_listof_outfile_paths(
                                roots, "cnmfe_cellset_out.isxd"
                            )
                        ),
                    )
                    try:
                        isx.longitudinal_registration(  # error: it thought meta_csv_filename was a movie input, so need the keyword
                            cellsets,
                            util.premake_listof_outfile_paths(
                                roots, "cnmfe_cellset_out.isxd"
                            ),
                            csv_file=meta_csv_filename,
                            min_correlation=0.50,
                            accepted_cells_only=True,
                        )
                        # print("here")

                        # preprocess long reg file and export
                        # GETTING THOSE CELLS ONLY ABOVE ncc score of above 0.9
                        mod_longreg_df = util.preprocess_longreg_results(
                            meta_csv_filename, 0.90, roots
                        )

                        mod_longreg_df.to_csv(
                            meta_csv_filename.replace(
                                "longreg_results.csv",
                                "longreg_results_preprocessed.csv",
                            ),
                            index=False,
                        )
                    except:
                        print("LONG REG COULD NOT BE PERFORMED IN THIS MOUSE")

                    # preprocess dff traces now and export
                    dff_file_paths = util.find_recursively("dff_traces.csv")
                    preprocessed_dff_filepaths = []
                    # the order of dff paths should correspond to that of roots, so
                    # the output should direct it to the right location

                    for i in range(len(dff_file_paths)):
                        # print(i)
                        df = pd.read_csv(dff_file_paths[i])
                        # print(df.head())
                        # print((df == " rejected").any()) does identify false rows to include
                        # print(df.columns)
                        # print(df.T.columns)
                        # print(list(df.T.columns.values.tolist())[1])
                        accepted_cells_dff = util.preprocess_dff_traces_csv(
                            dff_file_paths[i]
                        )
                        accepted_cells_dff.to_csv(
                            os.path.join(roots[i], "dff_traces_preprocessed.csv"),
                            index=False,
                        )
                        preprocessed_dff_filepaths.append(
                            os.path.join(roots[i], "dff_traces_preprocessed.csv")
                        )

                    """
                    Merge dff traces with longitudinal registration table:

                    for session name in session_name column:
                        find this folder, and in this folder, find the dff_preprocessed file,
                        if for this same row idx(for long reg csv) the cell name matches then
                        concatenate the dff traces for this cell onto cell reg

                    export this longreg file
                    - includes all sessions (but few cells), for each mice
                    """
                    # Find gpio file for the corresponding session of inscopix
                    "Finding GPIO time start and adding it onto ABET times"
                    # find list of gpio files, export to csv, open it as df
                    gpio_file_paths = util.find_raw_recursively(".gpio")
                    print("GPIOs found: ", gpio_file_paths)
                    for count, i in enumerate(gpio_file_paths):
                        root_out_path = roots[
                            count
                        ]  # gpio found based on root of cellsets path
                        print("Current working dir: ", root_out_path)
                        path_where_gpio_csv_saved = util.export_gpio(
                            i, root_to_out=root_out_path
                        )
                        # gpio is found in raw sessions, so get the csv
                        gpio_df = pd.read_csv(path_where_gpio_csv_saved)
                        gpio_start_time = util.get_time_gpio_starts(gpio_df)
                        print("gpio start time: ", gpio_start_time)

                        """Now can add on this gpio start time onto the threee columns in the corresponding
                        ABET data for this session: Start Time (s), Choice Time (s), Collection Time (s)
                        """
                        ABET_file_exists_in_dir = False
                        ABET_file_in_session = util.find_recursively_newroot(
                            root_out_path, "_ABET_processed.csv"
                        )
                        if len(ABET_file_in_session) != 0:
                            ABET_file_exists_in_dir = True
                            print("ABET file dir: ", ABET_file_in_session[0])
                            abet_added_gpio_df = util.add_gpio_start_time_to_ABET_cols(
                                ABET_file_in_session[0], gpio_start_time
                            )
                            print(abet_added_gpio_df.head())
                            abet_added_gpio_df.to_csv(
                                os.path.join(
                                    root_out_path,
                                    ABET_file_in_session[0].replace(
                                        "_ABET_processed.csv",
                                        "_ABET_GPIO_processed.csv",
                                    ),
                                ),
                                index=False,
                            )

                        else:
                            print(
                                ("Root path %s does not have an ABET file!")
                                % (root_out_path)
                            )

                    print("All done!")


if __name__ == "__main__":

    main()
    # run_per_mouse()


def run_per_mouse():
    ROOT_PATH = Path(r"/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-13")
    RAW_ROOT_PATH = Path(r"/media/rory/Nathen's Fantom/Inscopix_to_Analyze/BLA-Insc-13")

    # Indicate for what root directory are these utilities for
    util = CalciumPreprocessing(ROOT_PATH, RAW_ROOT_PATH)

    cellsets, roots = util.get_input_cell_set_files()
    # example: /media/rory/PTP Inscopix 2/PTP_Inscopix_#3/BLA-Insc-5/Session-20210510-093930_BLA-Insc-5_RM_D1/2021-05-10-09-45-37_video_BLA-Insc-5_RM_D1/cnmfe_cellset.isxd
    # session name is at the [6] split

    # Make sure output files don't already exist- else an error
    to_delete = [
        ".",
        "dff_traces.csv",
        "gpio.csv",
        "dff_traces_preprocessed_gpio.csv",
    ]
    print("Deleting files that contain the string(s): %s" % (to_delete))
    util.delete_recursively(to_delete)

    # Exporting csv dff traces to their corresponding session directories.
    for i in range(len(cellsets)):
        util.export_session_dff(
            cellsets[i],
            os.path.join(roots[i], "dff_traces.csv"),
            os.path.join(roots[i], "cell_"),
            "start",
            "",
        )

    # We don't need the tiff files that come with "export_session_dff()", so delete them
    to_delete = ["."]
    print("Deleting files that contain the string(s): %s" % (to_delete))
    util.delete_recursively(to_delete)

    # Perform long. reg. on the, but first, make cellset output files

    # do this deletion of out_files to make sure an error that it already exists does not appear
    to_delete = ["cnmfe_cellset_out.isxd", "longreg_results.csv"]
    print("Deleting files that contain the string(s): %s" % (to_delete))
    util.delete_recursively(to_delete)

    meta_csv_filename = os.path.join(util.root_path, "longreg_results.csv")

    print(
        len(cellsets),
        len(util.premake_listof_outfile_paths(roots, "cnmfe_cellset_out.isxd")),
    )
    try:
        isx.longitudinal_registration(  # error: it thought meta_csv_filename was a movie input, so need the keyword
            cellsets,
            util.premake_listof_outfile_paths(roots, "cnmfe_cellset_out.isxd"),
            csv_file=meta_csv_filename,
            min_correlation=0.50,
            accepted_cells_only=True,
        )
        # print("here")

        # preprocess long reg file and export
        mod_longreg_df = util.preprocess_longreg_results(meta_csv_filename, 0.90, roots)

        mod_longreg_df.to_csv(
            meta_csv_filename.replace(
                "longreg_results.csv", "longreg_results_preprocessed.csv"
            ),
            index=False,
        )
    except:
        print("LONG REG COULD NOT BE PERFORMED IN THIS MOUSE")

    # preprocess dff traces now and export
    dff_file_paths = util.find_recursively("dff_traces.csv")
    preprocessed_dff_filepaths = []
    # the order of dff paths should correspond to that of roots, so
    # the output should direct it to the right location

    for i in range(len(dff_file_paths)):
        # print(i)
        df = pd.read_csv(dff_file_paths[i])
        # print(df.head())
        # print((df == " rejected").any()) does identify false rows to include
        # print(df.columns)
        # print(df.T.columns)
        # print(list(df.T.columns.values.tolist())[1])
        accepted_cells_dff = util.preprocess_dff_traces_csv(dff_file_paths[i])
        accepted_cells_dff.to_csv(
            os.path.join(roots[i], "dff_traces_preprocessed.csv"), index=False
        )
        preprocessed_dff_filepaths.append(
            os.path.join(roots[i], "dff_traces_preprocessed.csv")
        )

    """
    Merge dff traces with longitudinal registration table:

    for session name in session_name column:
        find this folder, and in this folder, find the dff_preprocessed file,
        if for this same row idx(for long reg csv) the cell name matches then
        concatenate the dff traces for this cell onto cell reg

    export this longreg file
    - includes all sessions (but few cells), for each mice
    """
    # Find gpio file for the corresponding session of inscopix
    "Finding GPIO time start and adding it onto ABET times"
    # find list of gpio files, export to csv, open it as df
    gpio_file_paths = util.find_raw_recursively(".gpio")
    print("GPIOs found: ", gpio_file_paths)
    for count, i in enumerate(gpio_file_paths):
        root_out_path = roots[count]
        print("Current working dir: ", root_out_path)
        path_where_gpio_csv_saved = util.export_gpio(i, root_to_out=root_out_path)
        # gpio is found in raw sessions, so get the csv
        gpio_df = pd.read_csv(path_where_gpio_csv_saved)
        gpio_start_time = util.get_time_gpio_starts(gpio_df)
        print("gpio start time: ", gpio_start_time)

        """Now can add on this gpio start time onto the threee columns in the corresponding
        ABET data for this session: Start Time (s), Choice Time (s), Collection Time (s)
        """
        ABET_file_exists_in_dir = False
        ABET_file_in_session = util.find_recursively_newroot(
            root_out_path, "_ABET_processed.csv"
        )
        if len(ABET_file_in_session) != 0:
            ABET_file_exists_in_dir = True
            print("ABET file dir: ", ABET_file_in_session[0])
            abet_added_gpio_df = util.add_gpio_start_time_to_ABET_cols(
                ABET_file_in_session[0], gpio_start_time
            )
            print(abet_added_gpio_df.head())
            abet_added_gpio_df.to_csv(
                os.path.join(
                    root_out_path,
                    ABET_file_in_session[0].replace(
                        "_ABET_processed.csv", "_ABET_GPIO_processed.csv"
                    ),
                ),
                index=False,
            )

        else:
            print(("Root path %s does not have an ABET file!") % (root_out_path))

    print("All done!")
