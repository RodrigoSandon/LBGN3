import Utilities
from ISXTable import ISXTable


class Session:
    def __init__(self, id):

        self.id = id
        self.isx_csv = None
        self.dlc_csv = None
        self.behavioral_csv = None

    def get_session_id(self):
        return self.id()

    def add_isx_session(
        self, dff_csv_path, motioncorr_isxd, cellset_isxd, delete_cols=False
    ):

        """This is where you have a prompt that asks you the columns to omit and gives you the path to the file to open.
        Required: Have IDPS opened so that you just have to C/P the file path to view which columns you must open.
        """
        if delete_cols is False:
            print(
                "Open these two files in IDPS (open cnmfe_cellset file in motion_corrected file) to determine which cells to omit: ",
                motioncorr_isxd,
                " ",
                cellset_isxd,
            )

            cells_to_remove = Utilities.string_to_list(
                input(
                    "What are the cells (1<=) you'd like to omit? Separate values by commas, avoid spaces."
                )
            )

            print("Cells to remove: ", cells_to_remove)

        new_dff_table = ISXTable(
            dff_csv_path, cells_to_remove
        )  # add as and ISXTable obj

        self.isx_csv = new_dff_table

    def get_isx_session(self):
        return self.isx_csv

    def add_dlc_session(self, dlc_csv_path):
        return

    def add_behavioral_session(self, behavioral_csv_path):
        return
