import pandas as pd
import numpy as np
import os, glob

def timing_file_processing(file_path, fps):
    df = pd.read_csv(file_path)
    #iterate through columns except trial column
    for col in list(df.columns):
        if col != "Trial":
            time_format_to_sec(df, col, fps)

    #print(df.head())
    return df

def time_format_to_sec(df: pd.DataFrame, col, fps):
    new_lst = []
    old_lst = list(df[col])
    for i in old_lst:
        min = int(i.split(":")[0])
        sec = int(i.split(":")[1])
        min_to_sec = min * 60
        total_sec = min_to_sec + sec
        frame_num = total_sec * fps
        new_lst.append(frame_num)
    df[col] = new_lst


def freezing_output_processing(file_path):
    df_result = pd.read_csv(file_path)
    print(df_result[2690:2710])

    frame_list = list(df_result["Frame"])
    frame_list = [i + count for count, i in enumerate(frame_list)]
    df_result["Frame"] = frame_list

    print(df_result[2690:2710])
    return df_result

# do this after their processing
def freezing_alignment(df_freezing_out: pd.DataFrame, df_timing: pd.DataFrame):
    # add empty column first (zero-filled)
    df_freezing_out["Timestamps"] = [0] * len(df_freezing_out)
    replace_lst = list(df_freezing_out["Timestamps"])

    for col in list(df_timing.columns):
            
            if col == "CS ON":
                timestamps = list(df_timing[col])

                replace_func = lambda x: col if x in timestamps else x
                new_series = df_freezing_out["Frame"].apply(replace_func).tolist()

                #print(new_series)
                # now replace that old timestamps col with new_series
                for idx, val in enumerate(new_series):
                    if isinstance(val, str):
                        replace_lst[idx] = val
    print(df_freezing_out[2690:2710])

    df_freezing_out["Timestamps"] = replace_lst
    return df_freezing_out


def main():

    # experiment_types = ["Extinction", "Retrieval", "Conditioning"]
    experiment_type = ["Extinction", "Retrieval", "Conditioning"]
    

    ROOT_TIMING_FILE = "/media/rory/Padlock_DT/Fear_Conditioning_Control/"
    timing_file_name = f"{experiment_type}_CS_timing_FC_Control.csv"
    timing_filepath = os.path.join(ROOT_TIMING_FILE, timing_file_name)
    ROOT = f"/media/rory/Padlock_DT/Fear_Conditioning_Control/NewVideos/{experiment_type}"

    fps = 30

    file = "RRD276_Conditioning_FreezingOutput.csv"
    file_path = os.path.join(ROOT, file)
    file_out_path = file_path.replace(".csv", "_processed.csv")
    print(file_path)

    df = pd.read_csv(file_path)
    print(df[2690:2710])

    df_freezing_out = freezing_output_processing(file_path)
    df_timing = timing_file_processing(timing_filepath, fps)
    df_aligned = freezing_alignment(df_freezing_out, df_timing)

    df_aligned.to_csv(file_out_path, index=False)


if __name__ == "__main__":
    main()