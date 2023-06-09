import os
import math
import numpy as np
import pandas as pd
import EzTrackFunctions as ez

def find_file_with_strings(root_path, strings_list):
    strings_list = [s.lower() for s in strings_list]
    print(strings_list)
    for root, dirs, files in os.walk(root_path):
        for file in files:
            # ignore files that start with a dot (annoying lock files)
            if file.startswith("."):
                continue
            else:
                file_path = os.path.join(root, file).lower()
                if all(string.lower() in file_path for string in strings_list):
                    return os.path.join(root, file)
    return None

def time_to_frames(time, fps):
    time_str = str(time)
    if ":" in time_str:
        # Convert time in minutes to time in seconds
        minutes, seconds = map(int, time_str.split(":"))
        time_in_seconds = 60 * minutes + seconds
    else:
        time_in_seconds = int(time)

    # Calculate number of frames based on time and fps
    frames = int(time_in_seconds * fps)

    return frames

mouse_eztrack_path = "/media/rory/Padlock_DT/Fear_Conditioning_Control/Olena_Group/080922_extinction_1/VideoFreeze/hSyn-AS-Gi-3_FreezingOutput.csv"
mouse_eztrack_vid = "/media/rory/Padlock_DT/Fear_Conditioning_Control/Olena_Group/080922_extinction_1/VideoFreeze/hSyn-AS-Gi-3.avi"


mouse_eztrack_path_no_suffix = mouse_eztrack_path.replace(".csv","")
mouse_eztrack_path_processed = f"{mouse_eztrack_path_no_suffix}_processed_example.csv"

mouse_eztrack_df = pd.read_csv(mouse_eztrack_path)
PERSON_ROOT = "/media/rory/Padlock_DT/Fear_Conditioning_Control/Olena_Group"
person = "Olena"
experiment = "extinction_1"

fps_eztrack_adjusted = 30 # originally 60
correction_time = 4080
correction_time_in_frames = math.ceil(time_to_frames(correction_time, fps_eztrack_adjusted))


#finding the timing file for this person and experiment
# will search for according timing file based on the current directory that the
# freezing output file is in
timing_filepath = find_file_with_strings(PERSON_ROOT, [person.lower(), "FC_info", experiment])
df_timing = ez.timing_file_processing(timing_filepath, fps_eztrack_adjusted, correction_time_in_frames)
df_aligned = ez.freezing_alignment(mouse_eztrack_df, df_timing)

print("processed_freezing_out_path: ", mouse_eztrack_path_processed)
df_aligned.to_csv(mouse_eztrack_path_processed, index=False)


