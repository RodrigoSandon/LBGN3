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


class WilcoxonIdentityTest:
    def __init__(
        self,
        conn,
        cursor,
        db_name: str,
        df: pd.DataFrame,
        csv_path: str,
        session: str,
        event_type: str,
        base_lower_bound_time: int,
        base_upper_bound_time: int,
        lower_bound_time: int,
        upper_bound_time: int,
        reference_pair: dict,
        hertz: int,
        alpha,
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

        self.test = "mannwhitneyu"
        self.table_name = session

        self.base_lower_bound_time = base_lower_bound_time
        self.base_upper_bound_time = base_upper_bound_time
        self.lower_bound_time = lower_bound_time
        self.upper_bound_time = upper_bound_time
        self.reference_pair = reference_pair
        self.hertz = hertz

        self.alpha = alpha

        self.give_identity_wilcoxon()

    def make_col_name(self, sample_size):
        subwindow_base = f"{self.base_lower_bound_time}_to_{self.base_upper_bound_time}"
        subwindow_post = f"{self.lower_bound_time}_to_{self.upper_bound_time}"

        self.event_type = self.event_type.replace(" ", "_")
        lst = [
            self.test,
            self.event_type,
            str(sample_size),
            subwindow_base,
            subwindow_post,
        ]
        # SQL DOESNT LIKE THESE CHARACTERS
        key_name = "_".join(lst)
        if "." in key_name:
            key_name = key_name.replace(".", "dot")
        if "(" in key_name:
            key_name = key_name.replace("(", "")
        if ")" in key_name:
            key_name = key_name.replace(")", "")
        if "," in key_name:
            key_name = key_name.replace(",", "")
        if "'" in key_name:
            key_name = key_name.replace("'", "")
        if "-" in key_name:
            key_name = key_name.replace("-", "minus")
        return key_name

    def wilcoxon_rank_sum(self, number_cells, cell):

        sub_df_baseline_lst = create_subwindow_of_list(
            list(self.df[cell]),
            unknown_time_min=self.base_lower_bound_time,
            unknown_time_max=self.base_upper_bound_time,
            reference_pair=self.reference_pair,
            hertz=self.hertz,
        )

        sub_df_lst = create_subwindow_of_list(
            list(self.df[cell]),
            unknown_time_min=self.lower_bound_time,
            unknown_time_max=self.upper_bound_time,
            reference_pair=self.reference_pair,
            hertz=self.hertz,
        )

        if (sub_df_baseline_lst == sub_df_lst) == True:
            return "null"

        result_greater = stats.mannwhitneyu(
            sub_df_lst, sub_df_baseline_lst, alternative="greater"
        )

        result_less = stats.mannwhitneyu(
            sub_df_lst, sub_df_baseline_lst, alternative="less"
        )

        id = None
        if result_greater.pvalue < (self.alpha / number_cells):
            id = "+"
        elif result_less.pvalue < (self.alpha / number_cells):
            id = "-"
        else:
            id = "Neutral"

        return id

    def give_identity_wilcoxon(self):
        number_cells = len(list(self.df.columns))

        new_col_name = self.make_col_name(number_cells)
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
            id = self.wilcoxon_rank_sum(number_cells, cell)
            # print(id)

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
    db_name = "BLA_Cells_Ranksum_Post_Activity"

    # Create db connection
    conn = sqlite3.connect(f"{db_name}.db")
    c = conn.cursor()

    lst = os.listdir(ROOT)
    lst.reverse()
    for session in lst:
        print(session)
        SESSION_PATH = os.path.join(ROOT, session)

        csvs = find_paths_startswith(SESSION_PATH, "all_concat_cells.csv")

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
                WilcoxonIdentityTest(
                    conn,
                    c,
                    db_name,
                    change_cell_names(pd.read_csv(CONCAT_CELLS_PATH)),
                    CONCAT_CELLS_PATH,
                    session=table_name,
                    event_type="_".join(list_of_eventtype_name),
                    base_lower_bound_time=-10,
                    base_upper_bound_time=-5,
                    lower_bound_time=0,
                    upper_bound_time=3,
                    reference_pair={0: 100},
                    hertz=10,
                    alpha=0.01,
                )
            else:  # shock has a different structure, do accordingly
                # shock is 100 idxs long
                WilcoxonIdentityTest(
                    conn,
                    c,
                    db_name,
                    change_cell_names(pd.read_csv(CONCAT_CELLS_PATH)),
                    CONCAT_CELLS_PATH,
                    session=table_name,
                    event_type="_".join(list_of_eventtype_name),
                    base_lower_bound_time=-3,
                    base_upper_bound_time=0,
                    lower_bound_time=0,
                    upper_bound_time=3,
                    reference_pair={0: 50},
                    hertz=10,
                    alpha=0.01,
                )

    conn.close()


if __name__ == "__main__":
    main()
