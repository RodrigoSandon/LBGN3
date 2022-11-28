"""
Goals are the following:
    1) To extract h5 files from slp files
    2) Automate this conversion for 219 .slp files
    3) Extract this h5 file to its respective folder.

"""

from http.client import FOUND
from logging import root
import os
import glob
import h5py
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import pickle
import pandas as pd
import shutil
import cv2
import sys, subprocess

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path


def find_paths_endswith(root_path, endswith) -> list:

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )

    return files

def find_paths_endswith_recur_n_norecur(root_path, endswith) -> list:

    files_2 = []

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )

    # now if recursive is empty, then check in just the current tree
    if len(files) == 0:
        for i in os.listdir(root_path):
            if endswith in i:   
                files_2.append(os.path.join(root_path, i))
        
        return files_2
 
    return files


def find_path_no_middle_endswith_no_include_multiple(root_path, endswith, not_includes, not_includes2) -> list:
    found : str

    for i in os.listdir(root_path):
        if endswith in i and not_includes not in i and not_includes2 not in i:   
            found = os.path.join(root_path, i)
    return found

def find_path_no_middle_endswith(root_path, endswith, mouse) -> list:
    files = []

    for i in os.listdir(root_path):
        if endswith in i and mouse in i:   
            files.append(os.path.join(root_path, i))
    return files

def slp_file_parsing(slp_filename: str):
    slp_filename = slp_filename.split("/")[-1]
    mouse = slp_filename.split("_")[0]
    session = "_".join(slp_filename.split("_")[1:]).replace(
        ".mp4.predictions.slp", "")

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

def h5_file_parsing(slp_filename: str):
    slp_filename = slp_filename.split("/")[-1]
    mouse = slp_filename.split("_")[0]
    session = "_".join(slp_filename.split("_")[1:]).replace(
        ".mp4.predictions.h5", "")

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


def pix_per_frames_to_cm_per_s(i, fps):
    return i * (2.54/96) * (fps/1)


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


def export_sleap_data_mult_nodes(h5_filepath, session_root_path,mouse,session, fps):
    with h5py.File(h5_filepath, "r") as f:
        dset_names = list(f.keys())
        locations = fill_missing(f["tracks"][:].T)
        node_names = [n.decode() for n in f["node_names"][:]]

    for i, name in enumerate(node_names):
        if name == "body":
            print(f"Working on... {name}")
            # make a dir of this curr node
            node_folder = os.path.join(session_root_path, f"{mouse}_{session}_{name}")
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

def export_sleap_data_mult_nodes_body(h5_filepath, session_root_path, mouse,fps, session_type):
    with h5py.File(h5_filepath, "r") as f:
        dset_names = list(f.keys())
        locations = fill_missing(f["tracks"][:].T)
        node_names = [n.decode() for n in f["node_names"][:]]

    for i, name in enumerate(node_names):
        if name == "body":
            print(f"Working on... {name}")
            # make a dir of this curr node
            node_folder = os.path.join(session_root_path, f"{mouse}_{name}")
            os.makedirs(node_folder, exist_ok=True)
            os.chdir(node_folder)

            INDEX = i
            node_loc = locations[:, INDEX, :, :]

            vel_mouse = smooth_diff(node_loc[:, :, 0]).tolist()
            vel_mouse_to_cm_s = [pix_per_frames_to_cm_per_s(i, fps) for i in vel_mouse]

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
            step = float(1/fps)
            x_axis_time = np.arange(range_min, range_max, step).tolist()[:-1]
            print(len(x_axis_time))
            print(len([i for i in range(1, len(vel_mouse))]))
            print(len(x_coord_pix[:-1]))
            print(len(y_coord_pix[:-1]))
            print(len(x_coord_cm[:-1]))
            print(len(y_coord_cm[:-1]))
            print(len(vel_mouse[:-1]))
            print(len(vel_mouse_to_cm_s[:-1]))

            ######## EXPORTING CSV, COORD, VEL, TRACKS GRAPHS FOR EACH NODE ########
            csv_out = f"{mouse}_{session_type}_{name}_sleap_data.csv"
            export_to_csv(csv_out,
                        idx_time=x_axis_time,
                        idx_frame=[i for i in range(1, len(vel_mouse))],
                        x_pix=x_coord_pix[:-1],
                        y_pix=y_coord_pix[:-1],
                        x_cm=x_coord_cm[:-1],
                        y_cm=y_coord_cm[:-1],
                        vel_f_p=vel_mouse[:-1],
                        vel_cm_s=vel_mouse_to_cm_s[:-1])

            coord_vel_graphs_out = f"{mouse}_{session_type}_{name}_coord_vel.png"
            visualize_velocity_one_node(name,
                                        x_axis_time,
                                        x_coord_cm[:-1],
                                        y_coord_cm[:-1],
                                        vel_mouse[:-1],
                                        vel_mouse_to_cm_s[:-1],
                                        coord_vel_graphs_out)

            track_map_out = f"{mouse}_{session_type}_{name}_tracks.png"
            track_one_node(name, node_loc, track_map_out)

def get_frame_rate(filename):
    if not os.path.exists(filename):
        sys.stderr.write("ERROR: filename %r was not found!" % (filename,))
        return -1         
    out = subprocess.check_output(["ffprobe",filename,"-v","0","-select_streams","v","-print_format","flat","-show_entries","stream=avg_frame_rate"])
    rate = str(out).split('=')[1].replace("\"", "").replace("\'","").replace("\\n", "").split('/')
    #print(rate)
    if len(rate)==1:
        return float(rate[0])
    if len(rate)==2:
        return float(rate[0])/float(rate[1])
    return -1

def main():
    ROOT = r"/media/rory/RDT VIDS/BORIS_merge/BATCH_2"
    DST_ROOT = r"/media/rory/RDT VIDS/BORIS_merge/BATCH_2"

    slp_files = find_paths_endswith(ROOT, ".slp")

    """Other folders will go into a new root folder"""
    print("===== PROCESSING OTHER FILES =====")
    print(f"Number of SLP files: {len(slp_files)}")


    for count, j in enumerate(slp_files):
        slp_filename = j.split("/")[-1]
        mouse, session = slp_file_parsing(j)
        SESSION_ROOT = os.path.join(DST_ROOT, mouse, session)
        new_slp_path = os.path.join(SESSION_ROOT, slp_filename)
        movie = find_path_no_middle_endswith_no_include_multiple(SESSION_ROOT, "_merged_resized_grayscaled.mp4")
        print(f"movie found: {movie}")
        h5_path = new_slp_path.replace(".slp", ".h5")
        #UNCOMMENT FOR NEXT BATCH 8/12/2022
        #if os.path.exists(h5_path) == False:

        try:
            print(f"Processing {count + 1}/{len(slp_files)}")
            fps = get_frame_rate(movie)
            print(f"fps: {fps}")

            # 1) move the slp file
            os.makedirs(SESSION_ROOT, exist_ok=True)
            print(f"old path: {j} || new path: {new_slp_path}")

            shutil.copy(j, new_slp_path)

            # 2) Convert .slp to .h5
            slp_to_h5(new_slp_path, h5_path)

            # 3) Extract speed

            export_sleap_data_mult_nodes_body(h5_path, SESSION_ROOT, mouse,  fps)

        except Exception as e:
            print(e)
            pass

def main_just_extract_vel():
    ROOTS = [r"/media/rory/RDT VIDS/BORIS_merge/", r"/media/rory/RDT VIDS/BORIS/"]

    movie_reliable_end = "_merged_resized_grayscaled_reliable.mp4"
    movie_nonreliable_end = "_merged_resized_grayscaled.mp4"
    slp_end = ".predictions.slp"
    h5_end = ".predictions.h5"

    session_type = "choice"
    notsession_type = "outcome"

    # there can be two types of sleap data from each session type now
    # either w/ reliable label or no reliable label
    for ROOT in ROOTS:
        reliable_mice= []

        h5_files_1 = find_paths_endswith(ROOT, "_reliable.mp4.predictions.h5")
        for i in h5_files_1:
            mouse, session = h5_file_parsing(i)
            reliable_mice.append(mouse)
        
        # found reliables and recorded themevent
        new_h5_files_2 = []
        h5_files_2 = find_paths_endswith(ROOT, "_grayscaled.mp4.predictions.h5")
        # now filter nonreliable ones if the path includes any of the reliable mice
        for i in h5_files_2:
            mouse, session = h5_file_parsing(i)
            if mouse not in reliable_mice:
                # if it isn't already in h5_files_1 basically
                new_h5_files_2.append(i)
        
        # now add them together
        h5_files  = h5_files_1 + new_h5_files_2
        
        # now only get the ones of the session type you want
        h5_files = [i for i in h5_files if notsession_type not in i.lower()]

        print("===== PROCESSING H5 FILES =====")
        print(f"Number of H5 files: {len(h5_files)}")

        for count, h5_path in enumerate(h5_files):
            print(f"H5 file: {h5_path}")

            mouse, session = h5_file_parsing(h5_path)
            MOUSE_ROOT = os.path.join(ROOT, mouse.upper())

            movies = find_paths_endswith_recur_n_norecur(MOUSE_ROOT, movie_reliable_end)
            if len(movies) == 0:
                movies = find_paths_endswith_recur_n_norecur(MOUSE_ROOT, movie_nonreliable_end)

            # if movies is still empty after searching for both types of movis,
            # then do a search on root
            if len(movies) == 0:
                movies = find_path_no_middle_endswith(ROOT, movie_nonreliable_end, mouse)

            movies = [i for i in movies if notsession_type not in i.lower() and slp_end not in i and h5_end not in i]
            movie = movies[0]

            print(f"Movie found: {movie}")

            fps = get_frame_rate(movie)
            print(f"fps: {fps}")

            print(f"Processing {count + 1}/{len(h5_files)}")
            export_sleap_data_mult_nodes_body(h5_path, MOUSE_ROOT, mouse, fps, session_type)


def one_slp_file():
    
    slp_file_path = r"/media/rory/RDT VIDS/BORIS/RRD171/RRD171_RDT_OPTO_CHOICE_01042021_6_merged_resized_grayscaled_reliable.mp4.predictions.slp"
    # usually it's _merged_resized_grayscaled.mp4
    identifier = "_merged_resized_grayscaled_reliable.mp4"
    session_type = "choice"

    slp_filename = slp_file_path.split("/")[-1]
    mouse = slp_file_path.split("/")[5]

    DST_ROOT = slp_file_path.replace(slp_filename, "")
    SESSION_ROOT = slp_file_path.replace(slp_filename, "")

    new_slp_path = os.path.join(SESSION_ROOT, slp_filename)
    h5_path = new_slp_path.replace(".slp", ".h5")

    movie = find_path_no_middle_endswith_no_include_multiple(SESSION_ROOT, identifier, ".predictions.slp", ".predictions.h5")

    print(movie)
    fps = get_frame_rate(movie)
    print(f"fps: {fps}")
    slp_to_h5(new_slp_path, h5_path)

    export_sleap_data_mult_nodes_body(h5_path, SESSION_ROOT,mouse, fps, session_type)
    


if __name__ == "__main__":
    #main()
    #one_slp_file()
    main_just_extract_vel()
