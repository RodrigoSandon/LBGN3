import numpy as np
import pandas as pd
import os

csv = r"/Users/rodrigosandon/Documents/GitHub/LBGN3/SampleData/body_sleap_data.csv"
root = r"/Users/rodrigosandon/Documents/GitHub/LBGN3/SampleData/"


def process_speed_data(bodypart: str, root_dir: str, speed_filepath: str):
    def downsample(speed_filepath) -> pd.DataFrame:
        """Every 3 (after including the first one) rows, extract those rows and make a new df."""
        # if index == 0 or index % 3 == 0 --> keep as new df
        df = pd.read_csv(speed_filepath)

        indices_to_extract = []
        for idx, row in df.iterrows():
            if idx % 3 == 0:
                indices_to_extract.append(idx)

        sub_df = df.iloc[indices_to_extract]

        return sub_df

    def get_time_gpio_starts(gpio_path, stop_at=2) -> float:
        # at the second time you see "GPIO-1", that's when you choose the "Time (s)" value
        count = 0
        gpio_df = pd.read_csv(gpio_path)

        for row_idx in range(len(gpio_df)):
            if "GPIO-1" in gpio_df.loc[row_idx, " Channel Name"]:
                count += 1
                if count == stop_at:
                    time_ABET_starts = gpio_df.loc[row_idx, "Time (s)"]
                    print(f"GPIO correction value: {time_ABET_starts}")
                    return time_ABET_starts

        print("No start time found!")

    def gpio_correct_speed_data(root_dir: str, df: pd.DataFrame) -> pd.DataFrame:
        """There will be a gpio file in the same dir, jus find the value to add."""
        gpio_path = os.path.join(root_dir, "gpio.csv")

        gpio_correction = get_time_gpio_starts(gpio_path)  # start time of isx
        df["idx_time"] = float(df["idx_time"]) + float(gpio_correction)

        return df
    ##################################################################

    out_path = speed_filepath.replace(bodypart, f"processed_{bodypart}")

    sub_df = downsample(speed_filepath)
    print(f"length of downsampled df: {len(sub_df)}")

    gpio_corrected_df: pd.DataFrame
    gpio_corrected_df = gpio_correct_speed_data(root_dir, sub_df)
    gpio_corrected_df.to_csv(out_path, index=False)


process_speed_data("body", root, csv)
