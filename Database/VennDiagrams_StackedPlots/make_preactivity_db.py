import pandas as pd
from scipy import stats
import sqlite3
import glob, os

from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt


def find_paths_conditional_endswith(
    root_path, og_lookfor: str, cond_lookfor: str
) -> list:

    all_files = []

    for root, dirs, files in os.walk(root_path):

        if cond_lookfor in files:
            # acquire the trunc file
            file_path = os.path.join(root, cond_lookfor)
            # print(file_path)
            all_files.append(file_path)
        elif cond_lookfor not in files:
            # acquire the og lookfor
            file_path = os.path.join(root, og_lookfor)
            all_files.append(file_path)

    return all_files


def find_paths_startswith(root_path, startswith) -> list:

    files = glob.glob(
        os.path.join(root_path, "**", "%s*") % (startswith), recursive=True,
    )

    return files


def change_cell_names(df: pd.DataFrame):

    for col in df.columns:

        df = df.rename(columns={col: col.replace("BLA-Insc-", "")})
        # print(col)

    return df


def convert_secs_to_idx(
    unknown_time_min, unknown_time_max, reference_pair: dict, hertz: int
):
    reference_time = list(reference_pair.keys())[0]  # has to come from 0
    reference_idx = list(reference_pair.values())[0]

    idx_start = (unknown_time_min * hertz) + reference_idx

    idx_end = (unknown_time_max * hertz) + reference_idx  # exclusive
    return int(idx_start), int(idx_end)


def create_subwindow_of_list(
    lst, unknown_time_min, unknown_time_max, reference_pair, hertz
) -> list:
    idx_start, idx_end = convert_secs_to_idx(
        unknown_time_min, unknown_time_max, reference_pair, hertz
    )

    subwindow_lst = lst[idx_start:idx_end]
    return subwindow_lst


class IdentityGiver:
    def __init__(
        self,
        conn,
        cursor,
        db_name: str,
        df: pd.DataFrame,
        csv_path: str,
        session: str,
        event_type: str,
    ):
        #####SQL STUFF#####
        self.conn = conn
        self.cursor = cursor
        self.db_name = db_name
        #####SQL STUFF#####

        self.df = df
        self.csv_path = csv_path
        self.session = session
        self.event_type = event_type
        self.table_name = session

        self.give_identity()

    def make_col_name(self):

        self.event_type = self.event_type.replace(" ", "_")

        # SQL DOESNT LIKE THESE CHARACTERS
        
        if "." in self.event_type:
            self.event_type = self.event_type.replace(".", "dot")
        if "(" in self.event_type:
            self.event_type = self.event_type.replace("(", "")
        if ")" in self.event_type:
            self.event_type = self.event_type.replace(")", "")
        if "," in self.event_type:
            self.event_type = self.event_type.replace(",", "")
        if "'" in self.event_type:
            self.event_type = self.event_type.replace("'", "")
        if "-" in self.event_type:
            self.event_type = self.event_type.replace("-", "minus")
            
        return self.event_type

    def give_identity(self):

        new_col_name = self.make_col_name()
        # add test, comes before all the identity giving to cells
        print(new_col_name)
        self.cursor.execute(
            f"ALTER TABLE {self.table_name} ADD COLUMN {new_col_name} TEXT"
        )
        self.conn.commit()

        for cell in list(self.df.columns):

            # check if cell already exists, (not iin first run)
            result = None
            for row in self.cursor.execute(
                f"SELECT * FROM {self.table_name} WHERE {self.table_name}.cell_name = ?",
                (cell,),
            ):
                result = row

            # then add it's id value
            id = list(self.df[cell])[0]

            # cell doesn't exists: means we have an empty table
            if not isinstance(result, tuple):

                # insert cell name and id
                self.cursor.execute(
                    f"INSERT INTO {self.table_name} VALUES (?,?)", [cell, id]
                )
                self.conn.commit()

            # if cellalready exists: dont insert cell name, jus add test (new col) and its id val, must be a new subevent
            # (some data already exists in the db from first run)
            else:
                # now new to indicate where to put new value exactly

                self.cursor.execute(
                    f"UPDATE {self.table_name} SET {new_col_name}=(?) WHERE {self.table_name}.cell_name= (?)",
                    [id, cell],
                )
                self.conn.commit()


def main():
    ROOT = r"/media/rory/Padlock_DT/BLA_Analysis/BetweenMiceAlignmentData"

    # Set db name and curr subevent path
    db_name = "/media/rory/Padlock_DT/BLA_Analysis/Database/BLA_Cells_Pre_Activity.db"

    # Create db connection
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    lst = os.listdir(ROOT)
    lst.reverse()
    for session in lst:
        print(session)
        SESSION_PATH = os.path.join(ROOT, session)

        csvs = find_paths_startswith(SESSION_PATH, "all_concat_cells_z_fullwindow_id_auc_bonf0.05_pre.csv")

        # Create SQl table here
        table_name = session.replace(" ", "_")
        if "-" in table_name:
            table_name = table_name.replace("-", "_")

        c.execute(
            f"""

        CREATE TABLE {table_name} (
            cell_name TEXT
        )
        
        """
        )

        # IF table already created?

        for csv in csvs:
            # print(csv)
            CONCAT_CELLS_PATH = csv

            list_of_eventtype_name = [
                CONCAT_CELLS_PATH.split("/")[7],
                CONCAT_CELLS_PATH.split("/")[8],
            ]

            if "Shock Test" not in csv:

                # Run a test on a subevent
                # this is the cell identities so no more analysis, just placing
                IdentityGiver(
                    conn,
                    c,
                    db_name,
                    change_cell_names(pd.read_csv(CONCAT_CELLS_PATH)),
                    CONCAT_CELLS_PATH,
                    session=table_name,
                    event_type="_".join(list_of_eventtype_name),
                )
            else:  # shock has a different structure, do accordingly
                # shock is 100 idxs long
                IdentityGiver(
                    conn,
                    c,
                    db_name,
                    change_cell_names(pd.read_csv(CONCAT_CELLS_PATH)),
                    CONCAT_CELLS_PATH,
                    session=table_name,
                    event_type="_".join(list_of_eventtype_name),
                )

    conn.close()


if __name__ == "__main__":
    main()
