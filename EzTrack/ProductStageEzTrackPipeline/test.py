def convert_min_to_sec(min, sec):
    min_to_sec = int(min) * 60
    total_sec = min_to_sec + float(sec)
    return total_sec

def convert_sec_to_frames(sec, fps):
    frame_num = float(sec) * fps
    return frame_num

def break_down_timestamp(timestamp):
    """_summary_

    Args:
        timestamp (str): _description_

    Returns:
        int: _description_
        int: _description_
    """
    min = int(timestamp.split(":")[0])
    sec = int(timestamp.split(":")[1])
    return min, sec

def create_timestamps(start_timestamp: str, num_events: int, step: int, fps: float, units_steps, units_timestamp):
    """_summary_

    Args:
        start_timestamp (str): when to start the timestamps
        num_events (int): number of events
        step (int): step size, how many frames to skip
        fps (float): frames per second of video
        units_steps (str): the units of the step size
        units_timestamp (str): the units of the start timestamp

    Returns:
        list: list of timestamps
    """
    if units_timestamp == "min":
        min, sec = break_down_timestamp(start_timestamp)
        start_frame_num = convert_sec_to_frames(convert_min_to_sec(min, sec), fps)
    elif units_timestamp == "sec":
        start_frame_num = convert_sec_to_frames(start_timestamp, fps)

    if units_steps == "sec":
        step_in_frames  = convert_sec_to_frames(step, fps)
    elif units_steps == "min":
        min, sec = break_down_timestamp(step)
        step_in_frames = convert_sec_to_frames(convert_min_to_sec(min, sec), fps)   

    print("start_frame_num: ", start_frame_num)
    print("step_in_frames: ", step_in_frames)
    frame_stamps = [start_frame_num + i * step_in_frames for i in range(0, num_events)]
    print("frame_stamps: ", frame_stamps)
    return frame_stamps

create_timestamps(start_timestamp="3:00", num_events=3, step="1:30", fps=30, units_steps="min", units_timestamp="min")