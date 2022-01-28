import pandas as pd
import numpy as np


class ShockUtilities:
    def process_csv(df):
        """There's no boolean for whether it's shocked or not, just got shocked 26 times, know intensity,
        so just get times."""

        def shock_time():
            # print(df["Item_Name"])
            if any(df["Item_Name"].str.contains("Shock_Counter")):
                return df[df["Item_Name"].str.contains("Shock_Counter")][
                    "Evnt_Time"
                ].values[0]
            else:
                return "Not Found"

        return pd.Series({"Shock Time (s)": shock_time()})

    def del_first_row(df):
        df = df[1:]
        return df

    def add_shock_intensity(df):
        # print(df)
        new_col = np.arange(0, 0.52, 0.02).tolist()
        # print(f"New col: {new_col}")

        df.insert(1, "Shock Intensity (mA)", new_col)

        return df
