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
import cv2
import sys, subprocess

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path


def find_paths_endswith(root_path, endswith):

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )

    return files

def convert(string):
    list1 = []
    list1[:0] = string
    return list1

def slp_file_parsing(slp_filename: str):
    #print(slp_filename)
    mouse = slp_filename.split("_")[0]
    print(mouse)
    # this should skip mouse name and get up to the date
    session = "_".join(slp_filename.split("_")[1:5])

    return mouse, session

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


def export_sleap_data_mult_nodes(h5_filepath, session_root_path, mouse,session,fps):
    with h5py.File(h5_filepath, "r") as f:
        dset_names = list(f.keys())
        locations = fill_missing(f["tracks"][:].T)
        node_names = [n.decode() for n in f["node_names"][:]]

    for i, name in enumerate(node_names):
        print(f"Working on... {name}")
        # make a dir of this curr node
        node_folder = os.path.join(session_root_path, f"{mouse}_{session}_{name}")
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

        ######## EXPORTING CSV, COORD, VEL, TRACKS GRAPHS FOR EACH NODE ########
        csv_out = f"{mouse}_{session}_{name}_sleap_data.csv"
        print(csv_out)
        export_to_csv(csv_out,
                    idx_time=x_axis_time,
                    idx_frame=[i for i in range(1, len(vel_mouse))],
                    x_pix=x_coord_pix[:-1],
                    y_pix=y_coord_pix[:-1],
                    x_cm=x_coord_cm[:-1],
                    y_cm=y_coord_cm[:-1],
                    vel_f_p=vel_mouse[:-1],
                    vel_cm_s=vel_mouse_to_cm_s[:-1])

        coord_vel_graphs_out = f"{mouse}_{session}_{name}_coord_vel.png"
        visualize_velocity_one_node(name,
                                    x_axis_time,
                                    x_coord_cm[:-1],
                                    y_coord_cm[:-1],
                                    vel_mouse[:-1],
                                    vel_mouse_to_cm_s[:-1],
                                    coord_vel_graphs_out)


        track_map_out = f"{mouse}_{session}_{name}_tracks.png"
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
    ROOT = r"/media/rory/Padlock_DT/Test/" # change to windows root path
    DST_ROOT = r"/media/rory/Padlock_DT/Test/" # change to windows root path

    slp_files = find_paths_endswith(ROOT, ".slp")

    """Other folders will go into a new root folder"""
    print("===== PROCESSING OTHER FILES =====")
    print(f"Number of SLP files: {len(slp_files)}")

    for count, j in enumerate(slp_files):
        slp_filename = j.split("/")[-1]
        mouse, session = slp_file_parsing(slp_filename)
        SESSION_ROOT = os.path.join(DST_ROOT, mouse, session)
        new_slp_path = os.path.join(SESSION_ROOT, slp_filename)
        h5_path = new_slp_path.replace(".slp", ".h5")
        movie_filename = slp_filename.replace(".predictions.slp", "")
        movie_path = os.path.join(ROOT, movie_filename)
        new_movie_path = os.path.join(SESSION_ROOT, movie_filename)
        #UNCOMMENT FOR NEXT BATCH 8/12/2022
        #if os.path.exists(h5_path) == False:

        #try:
        print(f"Processing {count + 1}/{len(slp_files)}")
        fps = get_frame_rate(movie_path)
        print(f"fps: {fps}")


        # 1) move the slp file
        os.makedirs(SESSION_ROOT, exist_ok=True)

        print(f"old path: {j} || new path: {new_slp_path}")

        shutil.move(j, new_slp_path)
        # move the movie
        shutil.move(movie_path, new_movie_path)

        # 2) Convert .slp to .h5
        slp_to_h5(new_slp_path, h5_path)

        # 3) Extract speed
        #meta_data(h5_path)
        export_sleap_data_mult_nodes(h5_path, SESSION_ROOT, mouse,session, fps=30)
        #export_sleap_data_mult_nodes_body(h5_path, SESSION_ROOT,mouse, fps=30)
        """except Exception as e:
            print(e)
            pass"""


def one_slp_file():
    
    slp_file_path = "D:/SLEAP/Photometry/RRD256/RDT_D1/RRD256_RDT_D1_2022-09-10T11_27_27.avi.predictions.slp"
    
    slp_filename = slp_file_path.split("/")[-1]
    mouse = slp_file_path.split("/")[3] #change number in brackets to correspond to animal ID in folder structure
    DST_ROOT = slp_file_path.replace(slp_filename, "")
    SESSION_ROOT = slp_file_path.replace(slp_filename, "")
    new_slp_path = os.path.join(SESSION_ROOT, slp_filename)
    h5_path = new_slp_path.replace(".slp", ".h5")

    slp_to_h5(new_slp_path, h5_path)

    export_sleap_data_mult_nodes(h5_path, SESSION_ROOT,mouse, fps=30) #update FPS based on the video
    


if __name__ == "__main__":
    main()
    #one_slp_file()
