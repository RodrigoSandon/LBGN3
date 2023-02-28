Personal Variables:

    Below user sets `FreezeThresh`, the upper bound in frame-by-frame pixels changed for freezing to be detected, and `MinDuration`, 
    the number of frames motion must be below FreezeThresh to begin accruing freezing.

    1. FreezeThresh = 180 # the upper bound in frame-by-frame pixels changed for freezing to be detected
        - increase: more pixels needed to have change to have it considered freezing (stricter)
        - decrease: less pixels needed to have change to have it considered freezing (looser)

    2. MinDuration = 40 # number of frames motion must be below FreezeThresh to begin accruing freezing
        - increase: harder to detect freezing (stricter), increases the times it must considered freezing before it's
        actually considered freezing
        - decrease: easier to detect freezing (looser), decreases the times it must be considered freezing before it's
        actually considered freezing

    3. number_of_frames_to_calibrate = 600

    4. calibrate_video_what_frame_to_start = 0 

    5. vid_d_start = 0 # when to start freeze analysis on any given video

    6. event_tracked = 'CS ON'

    7. half_time_window = 30 # how much time after event occurred do you want to get? (in secs)

    8. fps = 30

    9. correspondence_filepath = "/media/rory/Padlock_DT/Fear_Conditioning_Control/mouse_chamber_corrrespondence.csv" # all mice

    10. experimental_groups_csv = "/media/rory/Padlock_DT/Fear_Conditioning_Control/experimental_groups.csv" # where is your file for corresponding opsins?

    11. ROOT_TIMING_FILE = "/media/rory/Padlock_DT/Fear_Conditioning_Control/" # where is the root of your timing files (for diff chambers)

    12. root_calibration_vids = f"/media/rory/Padlock_DT/Fear_Conditioning_Control/NewVideos/Calibration" # where are your cal vids located?

    13. ROOT = f"/media/rory/Padlock_DT/Fear_Conditioning_Control/NewVideos/" # of all vids from all exp types

    14. experiment_types = ["Extinction", "Retrieval", "Conditioning"] # there can be more if u'd like (or diff names)

    # other variable but that has to be done manually: calibration_vid_file = f"Chamber_{chamber}_calibration_extinction.avi"

Important Variables:

    1. FreezeThresh = 180 

    2. MinDuration = 40

    3. number_of_frames_to_calibrate = 600

    4. calibrate_video_what_frame_to_start = 0 

    5. vid_d_start = 0 # when to start freeze analysis on a given video

    6. event_tracked = 'CS ON'

    7. half_time_window = 30

    8. fps = 30

    9. correspondece_file = "mouse_chamber_corrrespondence.csv"

    10. colname_vid_paths = "mouse_vid_path"

    11. letter_column_name = "chamber"

    12. eztrack_output_processed_suffix = "FreezingOutput_processed.csv"

    13. experimental_groups_csv = "/media/rory/Padlock_DT/Fear_Conditioning_Control/experimental_groups.csv"

    14. ROOT_TIMING_FILE = "/media/rory/Padlock_DT/Fear_Conditioning_Control/"

    15. root_calibration_vids = f"/media/rory/Padlock_DT/Fear_Conditioning_Control/NewVideos/Calibration"

    16. ROOT = f"/media/rory/Padlock_DT/Fear_Conditioning_Control/NewVideos/"

    17. experiment_types = ["Extinction", "Retrieval", "Conditioning"]
    

Protocol:

1) Making your "mouse_chamber_correspondence.csv" files:

    Define a list of file paths for video files.

    For each file path in the list:

        a. Access the video file using the file path.

        b. Identify the chamber letter within the video file.

        c. Record the file path and identified letter in a table.

    The resulting table will have two columns: "mouse_vid_path" and "chamber" preferrably, where each 
    row corresponds to a single video file and the letter that was identified within that file.

2) Define where your calibration videos are, paste that into the variable: "root_calibration_vids".

3) Define where your root directory is, where all the videos are from all experiment types into "ROOT".

4) Define the root directory of where your timing files are (defines times for the event you're interested in), 
    paste into variable "ROOT_TIMING_FILE".

5) Define your timing like so on a csv file:

    +-------+-------+--------+---------------+----------------+
    | Trial | CS ON | CS OFF | US (SHOCK) ON | US (SHOCK) OFF |
    +-------+-------+--------+---------------+----------------+
    | 1     | 3:00  | 3:30   | 3:28          | 3:30           |
    +-------+-------+--------+---------------+----------------+
    | 2     | 4:30  | 5:00   | 4:58          | 5:00           |
    +-------+-------+--------+---------------+----------------+
    | 3     | 6:30  | 7:00   | 6:58          | 7:00           |
    +-------+-------+--------+---------------+----------------+

    Name it "{experiment_type}_CS_timing_FC_Control.csv".

6) It may be of interest to you to compare opsin groups performned in this experiment. For this, you're going to   
    to need to define a table that corressponds mice to their respective opsin groups like so:

    +-----------+-----------+
    | mouse_num | opsin     |
    +-----------+-----------+
    | 276       | ChrimsonR |
    +-----------+-----------+
    | 278       | ChrimsonR |
    +-----------+-----------+
    | 290       | ChrimsonR |
    +-----------+-----------+
    | 277       | mCherry   |
    +-----------+-----------+
    | 279       | mCherry   |
    +-----------+-----------+
    | 281       | mCherry   |
    +-----------+-----------+

    Name it "experimental_groups.csv" and define it's path and paste into the "experimental_groups_csv" variable.



    