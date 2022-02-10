import numpy as np
import pandas as pd
import os


def process_speed_data(bodypart: str, root_dir: str, speed_filepath: str):
    def downsample(df: pd.DataFrame) -> pd.DataFrame:
        """Every 3 (after including the first one) rows, extract those rows and make a new df."""
        # if index == 0 or index % 3 == 0 --> keep as new df
        indices_to_extract = []
        for idx, row in df.iterrows():
            if idx % 3 == 0:
                indices_to_extract.append(idx)

        sub_df = df.iloc[indices_to_extract]

        print(len(sub_df))

        return sub_df

    def binary_search(data: list, val):
        """Will return index if the value is found, otherwise the index of the item that is closest
        to that value."""
        lo, hi = 0, len(data) - 1
        best_ind = lo
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if data[mid] < val:
                lo = mid + 1
            elif data[mid] > val:
                hi = mid - 1
            else:
                best_ind = mid
                break
            # check if data[mid] is closer to val than data[best_ind]
            if abs(data[mid] - val) < abs(data[best_ind] - val):
                best_ind = mid

        return best_ind

    def get_time_gpio_starts(gpio_path, stop_at=2) -> float:
        # at the second time you see "GPIO-1", that's when you choose the "Time (s)" value
        count = 0
        gpio_df = pd.read_csv(gpio_path)

        for row_idx in range(len(gpio_df)):
            if "GPIO-1" in gpio_df.loc[row_idx, " Channel Name"]:
                count += 1
                if count == stop_at:
                    time_ABET_starts = gpio_df.loc[row_idx, "Time (s)"]
                    print(f"GPIO correction: {time_ABET_starts}")
                    return time_ABET_starts

        print("No start time found!")

    def gpio_correct_speed_data(root_dir: str, df: pd.DataFrame) -> pd.DataFrame:
        """There will be a gpio file in the same dir, jus find the value to add."""
        gpio_path = os.path.join(root_dir, "gpio.csv")

        gpio_correction = get_time_gpio_starts(gpio_path)  # start time of isx
        df["idx_time"] = df["idx_time"] + float(gpio_correction)

        return df

    def zerolistmaker(n):
        listofzeros = [0] * n
        return listofzeros

    def insert_sleap_into_dff_data(root_dir: str, sleap_df: pd.DataFrame) -> pd.DataFrame:

        dff_path = os.path.join(root_dir, "dff_traces_preprocessed.csv")
        dff_df = pd.read_csv(dff_path)
        vel_cm_s_list = zerolistmaker(len(dff_df))

        sleap_timestamps = list(sleap_df["idx_time"])
        dff_timestamps = list(dff_df["Time(s)"])
        start_of_sleap = sleap_timestamps[0]
        end_of_sleap = sleap_timestamps[-1]
        dff_timestamps_start_idx = binary_search(
            dff_timestamps, start_of_sleap)
        dff_timestamps_stop_idx = binary_search(dff_timestamps, end_of_sleap)

        dff_timestamps_lookfor = dff_timestamps[
            dff_timestamps_start_idx: dff_timestamps_stop_idx + 1]

        # no need to truncate original dff traces timestamps to help out algo?
        for count, dff_timestamp in enumerate(dff_timestamps_lookfor):
            closest_sleap_timestamp_idx_match_to_dff_trace = binary_search(
                sleap_timestamps, dff_timestamp)
            idx = closest_sleap_timestamp_idx_match_to_dff_trace

            if count % 500 == 0:  # just so it won't print a lot
                print(
                    f"Df/f timestamp: {dff_timestamp} | SLEAP timestamp: {sleap_timestamps[idx]}")
            # ^ returns idx of sleap "idx_time" col
            # Get all col values in which pertain to this sleap timestamp
            # Only change curr "idx_time" to the current dff_timestamp
            # this should produce a dff of the same length as the dff_traces_preprocessed.csv
            # val needs to match that of vel_cm_s_list(from index) and dff_f
            """print(dff_df.index[dff_df["Time(s)"] == dff_timestamp].tolist()[0])
            print(type(dff_df.index[dff_df["Time(s)"]
                                    == dff_timestamp].tolist()[0]))"""
            vel_cm_s_list[dff_df.index[dff_df["Time(s)"] == dff_timestamp].tolist()[
                0]] = sleap_df.iloc[idx]["vel_cm_s"]

        # check lengths of dff and new col that you're gonna input
        print(
            f"Length of dff traces: {len(dff_df)} | Length of sleap data: {dff_timestamps_stop_idx - dff_timestamps_start_idx}")
        # Now should have list ready to insert
        dff_df.insert(1, "Speed(cm/s)", vel_cm_s_list)

        return dff_df

    ##################################################################
    sleap_df = pd.read_csv(speed_filepath)
    out_path = speed_filepath.replace(bodypart, f"dff_and_{bodypart}")

    # sub_df = downsample(speed_filepath)
    # print(f"length of downsampled df: {len(sub_df)}")

    gpio_corrected_df: pd.DataFrame
    gpio_corrected_df = gpio_correct_speed_data(root_dir, sleap_df)
    gpio_corrected_df.to_csv(speed_filepath.replace(
        bodypart, f"gpio_corrected_{bodypart}"), index=False)

    print("here")
    new_df = insert_sleap_into_dff_data(root_dir, gpio_corrected_df)
    print("here")

    new_df.to_csv(out_path, index=False)


csv = "/Users/rodrigosandon/Documents/GitHub/LBGN3/SampleData/body_sleap_data.csv"
root = "/Users/rodrigosandon/Documents/GitHub/LBGN3/SampleData/"

process_speed_data("body", root, csv)
