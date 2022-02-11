"""
Goals are the following:
    1) To extract h5 files from slp files
    2) Automate this conversion for 219 .slp files
    3) Extract this h5 file to its respective folder.

"""

import os
import glob
import h5py
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import pickle
import pandas as pd
import shutil

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


def find_paths_endswith(root_path, endswith) -> list:

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )

    return files


def slp_file_parsing(slp_filename: str):
    slp_filename = slp_filename.split("/")[-1]
    mouse = slp_filename.split("_")[0]
    session = "_".join(slp_filename.split("_")[1:]).replace(
        ".avi.predictions.slp", "")

    try:
        session_mod_1 = session.split("20")[0]

        if "." in session_mod_1:
            session_mod_1 = session_mod_1.replace(".", "")
        if "_" in session_mod_1:
            session_mod_1 = session_mod_1.replace("_", " ")

        #print(f"{mouse}: {session_mod_1}")

        return mouse, session_mod_1
    except Exception as e:
        print(f"Session {mouse}: {session} can't be renamed!")
        print(e)
        pass

# e1: /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-8/RDT D1/Session-20211021-093007_BLA-INSC-8-RDT-D1
# e2: /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1 <- (bla1-3)
# structure1: /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#{}/BLA-Insc-{}/{}
# structure2: /media/rory/Padlock_DT/BLA_Analysis/{}/BLA-Insc-{}/{}/{}
#                                     {if blax-x -> pertain to this ptp}    {y}{y}{startswith "session"}/->GO

    # 1) Find ptp folder
def ptp_autoencoder(mouse_tolower: str) -> str:
    """Gives you a ptp #, based on mouse name."""
    mouse_tolower = mouse_tolower.lower()
    d = {
        "PTP_Inscopix_#1": ["bla-insc-1", "bla-insc-2", "bla-insc-3"],
        "PTP_Inscopix_#3": ["bla-insc-5", "bla-insc-6", "bla-insc-7"],
        "PTP_Inscopix_#4": ["bla-insc-8", "bla-insc-9", "bla-insc-11", "bla-insc-13"]
    }

    for key in d.keys():
        if mouse_tolower in d[key]:
            return key

def find_dst_for_mouse_and_session(base, mouse: str, session: str) -> str:
    # base = /media/rory/Padlock_DT/BLA_Analysis/

    # 2) Find the appropriate mouse #, since capitalization is diff in the dst dir
    def mouse_name_autoencoder(mouse) -> int:
        """Gives you a number based on the number that's in the input."""
        num = int(mouse.split("-")[-1])
        # print(num)
        return num

    # 3) Session str is already properly formatted, nothing to do here.
    # 4) Need to find the interfolder (only exists for bla5-13)
    def add_interfolder(curr_dir) -> str:
        # So far have this: /media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-8/RDT D1/
        
        dir_found = None
        for dir in os.listdir(curr_dir):
            # if the substrings that identified a interfolder are present, then add that to dst
            print(dir)
            if dir.find("-") != -1:
                dir_found = dir
        
        if dir_found == None:
            print("CAN'T FIND INTERFOLDER!!")

        return dir_found

    ptp_folder = ptp_autoencoder(mouse)

    # Add ptp folder specification
    dst = os.path.join(base, ptp_folder)
    # Add mouse folder specification
    dst = os.path.join(
        dst, f"BLA-Insc-{mouse_name_autoencoder(mouse)}")
    # Add session type specification
    dst = os.path.join(dst, session)

    # Because only the other PTP folders have interfolders
    if ptp_folder != "PTP_Inscopix_#1":
        dst = os.path.join(dst, add_interfolder(dst))

    # print(dst)
    return dst


def create_bla_dst_path(base, file):
    # gets the mouse and session
    mouse, session = slp_file_parsing(file)
    #mouse = mouse.lower()
    #session = session.lower()
    dst = find_dst_for_mouse_and_session(base, mouse, session)
    #print(f"SLP destination for {mouse} {session}: {dst}")

    return dst


def send_all_other_files_somewhere(other_slp_files: list):
    for i in other_slp_files:
        slp_file_parsing(i)


def slp_to_h5(in_path, out_path):
    if " " in in_path:
            in_path = in_path.replace(" ", "\ ")
    if " " in out_path:
            out_path = out_path.replace(" ", "\ ")
    cmd = f"sleap-convert --format analysis -o {out_path} {in_path}"
    os.system(cmd)


def fill_missing(Y, kind="linear"):
    """Fills missing values independently along each dimension after the first."""

    # Store initial shape.
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)

        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(
            mask), np.flatnonzero(~mask), y[~mask])

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y


def meta_data(h5_filename):

    with h5py.File(h5_filename, "r") as f:
        dset_names = list(f.keys())
        locations = fill_missing(f["tracks"][:].T)
        node_names = [n.decode() for n in f["node_names"][:]]

    print("===filename===")
    print(h5_filename)

    print("===HDF5 datasets===")
    print(dset_names)

    print("===locations data shape===")
    print(locations.shape)

    print("===nodes===")
    for i, name in enumerate(node_names):
        print(f"{i}: {name}")


def smooth_diff(node_loc, win=25, poly=3):
    """
    node_loc is a [frames, 2] array

    win defines the window to smooth over

    poly defines the order of the polynomial
    to fit with

    """
    node_loc_vel = np.zeros_like(node_loc)

    for c in range(node_loc.shape[-1]):
        node_loc_vel[:, c] = savgol_filter(node_loc[:, c], win, poly, deriv=1)

    node_vel = np.linalg.norm(node_loc_vel, axis=1)

    return node_vel


def pix_to_cm(i):
    # There are 96 pix in 2.54 cm
    return i * (2.54/96)


def pix_per_frames_to_cm_per_s(i):
    return i * (2.54/96) * (30/1)


def export_to_csv(out_path, **kwargs):

    df = pd.DataFrame.from_dict(kwargs)

    df.to_csv(out_path, index=False)


def track_one_node(node_name, node_loc, track_map_out):

    plt.figure(figsize=(7, 7))
    plt.plot(node_loc[:, 0, 0], node_loc[:, 1, 0], 'y', label='mouse-0')
    plt.legend()

    plt.xlim(0, 1024)
    plt.xticks([])

    plt.ylim(0, 1024)
    plt.yticks([])
    plt.title(f'{node_name} tracks')
    plt.savefig(track_map_out)
    plt.close()


""""Will no longer call export to csv, another func will do that."""


def visualize_velocity_one_node(node_name, x_axis_time, x_coord_cm, y_coord_cm, vel_mouse, vel_mouse_to_cm_s, coord_vel_graphs_out):

    fig = plt.figure(figsize=(15, 7))
    fig.tight_layout(pad=10.0)

    ax1 = fig.add_subplot(211)
    # the format is (x,y,**kwargs)
    ax1.plot(x_axis_time, x_coord_cm, 'k', label='x')
    ax1.plot(x_axis_time, y_coord_cm, 'r', label='y')
    ax1.legend()

    ax1.set_xticks([i for i in range(0, len(vel_mouse), 600)])

    ax1.set_title(f'{node_name} X-Y Dynamics')
    ax1.set_ylabel("Coordinate (cm)")

    ax2 = fig.add_subplot(212, sharex=ax1)
    """For getting a heatmap version of the velocity."""
    #ax2.imshow(body_vel_mouse[:,np.newaxis].T, aspect='auto', vmin=0, vmax=10)

    ax2.plot(x_axis_time, vel_mouse_to_cm_s, label='Forward Speed')
    ax2.set_yticks([i for i in range(0, 28, 4)])
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Speed (cm/s)")
    # ax2.legend()

    plt.savefig(coord_vel_graphs_out)
    plt.close()


"""Will call on export to csv for every node"""


def export_sleap_data_mult_nodes(h5_filepath, session_root_path, fps):
    with h5py.File(h5_filepath, "r") as f:
        dset_names = list(f.keys())
        locations = fill_missing(f["tracks"][:].T)
        node_names = [n.decode() for n in f["node_names"][:]]

    for i, name in enumerate(node_names):
        print(f"Working on... {name}")
        # make a dir of this curr node
        node_folder = os.path.join(session_root_path, name)
        os.makedirs(node_folder, exist_ok=True)
        os.chdir(node_folder)

        INDEX = i
        node_loc = locations[:, INDEX, :, :]

        vel_mouse = smooth_diff(node_loc[:, :, 0]).tolist()
        vel_mouse_to_cm_s = [pix_per_frames_to_cm_per_s(i) for i in vel_mouse]

        fig = plt.figure(figsize=(15, 7))
        fig.tight_layout(pad=10.0)

        x_coord_pix = node_loc[:, 0, 0]

        y_coord_pix = node_loc[:, 1, 0]

        x_coord_cm = [(pix_to_cm(i)) for i in node_loc[:, 0, 0].tolist()]

        y_coord_cm = [(pix_to_cm(i)) for i in node_loc[:, 1, 0].tolist()]

        range_min = 0
        range_max = len(vel_mouse)/fps

        # step is 30 Hz, so 0.033 s in 1 frame
        # step = float(1/fps) <-- this should almost work, a rounding issue (this x_axis is one off of all other arrays that are going to be made into df)
        step = 0.03333333
        x_axis_time = np.arange(range_min, range_max, step).tolist()[:-1]

        ######## EXPORTING CSV, COORD, VEL, TRACKS GRAPHS FOR EACH NODE ########
        csv_out = f"{name}_sleap_data.csv"
        export_to_csv(csv_out,
                      idx_time=x_axis_time,
                      idx_frame=[i for i in range(1, len(vel_mouse)+1)],
                      x_pix=x_coord_pix,
                      y_pix=y_coord_pix,
                      x_cm=x_coord_cm,
                      y_cm=y_coord_cm,
                      vel_f_p=vel_mouse,
                      vel_cm_s=vel_mouse_to_cm_s)

        coord_vel_graphs_out = f"{name}_coord_vel.png"
        visualize_velocity_one_node(name,
                                    x_axis_time,
                                    x_coord_cm,
                                    y_coord_cm,
                                    vel_mouse,
                                    vel_mouse_to_cm_s,
                                    coord_vel_graphs_out)

        track_map_out = f"{name}_tracks.png"
        track_one_node(name, node_loc, track_map_out)


def merge_isx_and_speed_data(bodypart: str, root_dir: str, speed_filepath: str):
    def downsample(df: pd.DataFrame) -> pd.DataFrame:
        """Every 3 (after including the first one) rows, extract those rows and make a new df."""
        # if index == 0 or index % 3 == 0 --> keep as new df
        indices_to_extract = []
        for idx, row in df.iterrows():
            if idx % 3 == 0:
                indices_to_extract.append(idx)

        sub_df = df.iloc[indices_to_extract]

        #print(len(sub_df))

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

            """if count % 500 == 0:  # just so it won't print a lot
                print(
                    f"Df/f timestamp: {dff_timestamp} | SLEAP timestamp: {sleap_timestamps[idx]}")"""
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
    out_path = os.path.join(root_dir, f"processed_dff_and_{bodypart}_data.csv")

    # sub_df = downsample(speed_filepath)
    # print(f"length of downsampled df: {len(sub_df)}")

    gpio_corrected_df: pd.DataFrame
    gpio_corrected_df = gpio_correct_speed_data(root_dir, sleap_df)
    gpio_corrected_df_path = os.path.join(root_dir, bodypart, f"gpio_corrected_{bodypart}_sleap_data.csv")
    gpio_corrected_df.to_csv(gpio_corrected_df_path, index=False)

    print("here")
    new_df = insert_sleap_into_dff_data(root_dir, gpio_corrected_df)
    print("here")

    new_df.to_csv(out_path, index=False)


def main():
    ROOT = r"/media/rory/Padlock_DT/DeepLabCut_RDT_Sessions_Only"
    BLA_DST_ROOT = r"/media/rory/Padlock_DT/BLA_Analysis"
    OTHER_DST_ROOT = r"/media/rory/Padlock_DT/Non-BLA_Analysis"

    slp_files = find_paths_endswith(ROOT, ".slp")

    bla_slp_files = [i for i in slp_files if "BLA" in i]
    other_slp_files = [i for i in slp_files if "BLA" not in i]

    print("===== PROCESSING BLA FILES =====")
    print(f"Number of BLA SLP files: {len(bla_slp_files)}")

    # Will be actually putting the bla's into existing folders
    for count, i in enumerate(bla_slp_files):
        if count >= 25:
            print(f"Processing {count + 1}: {i}")
            # 1) move the slp file
            slp_filename = i.split("/")[-1]
            try:
                ses_root = create_bla_dst_path(BLA_DST_ROOT, i)
                new_slp_path = os.path.join(ses_root, slp_filename)
                print(f"old path: {i} || new path: {new_slp_path}")
                shutil.copy(i, new_slp_path)
            #videos in which make session that don't exists for BLA
            except FileNotFoundError as e:
                print(e)
                print("THIS SESSION DOESN'T CONTAIN ABET NOR ISX DATA")
                mouse, session = slp_file_parsing(i)
                ptp = ptp_autoencoder(mouse)
                ses_root = os.path.join(BLA_DST_ROOT, ptp, mouse, session)
                new_slp_path = os.path.join(ses_root, slp_filename)
                os.makedirs(ses_root, exist_ok=True)
                shutil.copy(i, new_slp_path)

            # 2) Convert .slp to .h5
            h5_path = new_slp_path.replace(".slp", ".h5")
            slp_to_h5(new_slp_path, h5_path)

            # 3) Extract speed
            #meta_data(h5_path)
            export_sleap_data_mult_nodes(h5_path, ses_root, fps=30)

            # 4) Preprocess the speed file

            bodypart = "body"
            speed_filepath = os.path.join(
                ses_root, bodypart, f"{bodypart}_sleap_data.csv")

            try:
                merge_isx_and_speed_data(bodypart, ses_root, speed_filepath)
                # ^ will EXPORT a dff_and_{bodypart}_sleap_data.csv and gpio corrected sleap data file"""
            except FileNotFoundError as e:
                print(e)
                pass

    """Other folders will go into a new root folder"""
    print("===== PROCESSING OTHER FILES =====")
    print(f"Number of OTHER SLP files: {len(other_slp_files)}")

    # IF THIS STOPS THEN ONLY RUN THIS PART AGAIN, DONT RERUN IF ALL BLA WAS SUCCESSUL
    for count2, j in enumerate(other_slp_files):
        print(f"Processing {count2 + 1}: {j}")
        # 1) move the slp file
        slp_filename = j.split("/")[-1]
        mouse, session = slp_file_parsing(j)
        SESSION_ROOT = os.path.join(OTHER_DST_ROOT, mouse, session)
        new_slp_path = os.path.join(SESSION_ROOT, slp_filename)
        os.makedirs(SESSION_ROOT, exist_ok=True)
        print(f"old path: {j} || new path: {new_slp_path}")

        shutil.copy(j, new_slp_path)

        # 2) Convert .slp to .h5
        h5_path = new_slp_path.replace(".slp", ".h5")
        slp_to_h5(new_slp_path, h5_path)

        # 3) Extract speed
        #meta_data(h5_path)
        export_sleap_data_mult_nodes(h5_path, SESSION_ROOT, fps=30)



if __name__ == "__main__":
    main()
