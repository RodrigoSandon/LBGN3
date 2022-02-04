
import os
import glob
import h5py
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import pickle
import pandas as pd

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

def slp_to_h5(in_path, out_path):

    cmd = f"sleap-convert --format analysis -o {out_path} {in_path}"
    os.system(cmd)

def pix_to_cm(i):
    # There are 96 pix in 2.54 cm
    return i * (2.54/96)
def pix_per_frames_to_cm_per_s(i):
    return i * (2.54/96) * (30/1)

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
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y

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
    
    node_vel = np.linalg.norm(node_loc_vel,axis=1)

    return node_vel

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

    frame_count, node_count, _, instance_count = locations.shape

    print("frame count:", frame_count)
    print("node count:", node_count)
    print("instance count:", instance_count)
    print()

def track_one_node(node_name, node_loc, track_map_out):

    plt.figure(figsize=(7,7))
    plt.plot(node_loc[:,0,0],node_loc[:,1,0], 'y',label='mouse-0')
    plt.legend()

    plt.xlim(0,1024)
    plt.xticks([])

    plt.ylim(0,1024)
    plt.yticks([])
    plt.title(f'{node_name} tracks')
    plt.savefig(track_map_out)

""""Will no longer call export to csv, another func will do that."""
def visualize_velocity_one_node(node_name, x_axis_time,x_coord_cm, y_coord_cm, vel_mouse, vel_mouse_to_cm_s, coord_vel_graphs_out):
    
    fig = plt.figure(figsize=(15,7))
    fig.tight_layout(pad=10.0)


    ax1 = fig.add_subplot(211)
    # the format is (x,y,**kwargs)
    ax1.plot(x_axis_time, x_coord_cm, 'k', label='x')
    ax1.plot(x_axis_time, y_coord_cm, 'r', label='y')
    ax1.legend()

    ax1.set_xticks([i for i in range(0,len(vel_mouse), 600)])

    ax1.set_title(f'{node_name} X-Y Dynamics')
    ax1.set_ylabel("Coordinate (cm)")

    ax2 = fig.add_subplot(212, sharex=ax1)
    """For getting a heatmap version of the velocity."""
    #ax2.imshow(body_vel_mouse[:,np.newaxis].T, aspect='auto', vmin=0, vmax=10)

    ax2.plot(x_axis_time, vel_mouse_to_cm_s, label='Forward Speed')
    ax2.set_yticks([i for i in range(0,28, 4)])
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Speed (cm/s)")
    #ax2.legend()

    plt.savefig(coord_vel_graphs_out)


"""Will call on export to csv for every node"""
def export_sleap_data_mult_nodes(h5_filepath, session_root_path, fps):
    with h5py.File(h5_filepath, "r") as f:
        dset_names = list(f.keys())
        locations = fill_missing(f["tracks"][:].T)
        node_names = [n.decode() for n in f["node_names"][:]]

    for i, name in enumerate(node_names):
        print(f"Working on... {name}")
        #make a dir of this curr node
        node_folder = os.path.join(session_root_path, name)
        os.makedirs(node_folder, exist_ok=True)
        os.chdir(node_folder)

        INDEX = i
        node_loc = locations[:, INDEX, :, :]

        vel_mouse = smooth_diff(node_loc[:, :, 0]).tolist()
        vel_mouse_to_cm_s = [pix_per_frames_to_cm_per_s(i) for i in vel_mouse]
        
        fig = plt.figure(figsize=(15,7))
        fig.tight_layout(pad=10.0)
        
        x_coord_pix = node_loc[:, 0, 0]

        y_coord_pix = node_loc[:, 1, 0]

        x_coord_cm = [(pix_to_cm(i)) for i in node_loc[:, 0, 0].tolist()]

        y_coord_cm = [(pix_to_cm(i)) for i in node_loc[:, 1, 0].tolist()]

        range_min = 0
        range_max = len(vel_mouse)/fps

        #step is 30 Hz, so 0.033 s in 1 frame
        #step = float(1/fps) <-- this should almost work, a rounding issue (this x_axis is one off of all other arrays that are going to be made into df)
        step = 0.03333333
        x_axis_time = np.arange(range_min, range_max, step).tolist()[:-1]

        ######## EXPORTING CSV, COORD, VEL, TRACKS GRAPHS FOR EACH NODE ########
        csv_out = f"{name}_sleap_data.csv"
        export_to_csv(csv_out,
                    idx_time=x_axis_time,
                    idx_frame=[i for i in range(1,len(vel_mouse)+1)],
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

def export_to_csv(out_path, **kwargs):

    df = pd.DataFrame.from_dict(kwargs)

    df.to_csv(out_path, index=False)
    
def main():
    MOUSE_ROOT = r"/media/rory/Padlock_DT/Velocity/BLA-Insc-8"
    SESSION_ROOT = os.path.join(MOUSE_ROOT, "RDT_D1_newpred")
    slp_filepath = os.path.join(SESSION_ROOT, "BLA-Insc-8_RDT_D12021-10-21T11_18_16.avi.220204_142416.predictions.slp")
    h5_filepath = os.path.join(SESSION_ROOT, "BLA-Insc-8_RDT_D12021-10-21T11_18_16.avi.220204_142416.predictions.h5")
    slp_to_h5(slp_filepath, h5_filepath)

    meta_data(h5_filepath)
    export_sleap_data_mult_nodes(h5_filepath, SESSION_ROOT, fps=30)



if __name__ == "__main__":
    main()