
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

def pix_to_cm():
    pass

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

def visualize_movement(locations):
    BODY_INDEX = 4

    body_loc = locations[:, BODY_INDEX, :, :]

    sns.set('notebook', 'ticks', font_scale=1.2)
    mpl.rcParams['figure.figsize'] = [15,6]

    plt.figure()
    plt.plot(body_loc[:,0,0], 'y',label='mouse-0')

    plt.legend(loc="center right")
    plt.title('Body locations')
    plt.show()

    #######

    plt.figure(figsize=(7,7))
    plt.plot(body_loc[:,0,0],body_loc[:,1,0], 'y',label='mouse-0')
    plt.legend()

    plt.xlim(0,1024)
    plt.xticks([])

    plt.ylim(0,1024)
    plt.yticks([])
    plt.title('Body tracks')
    plt.show()

def meta_data(h5_filename):

    with h5py.File(h5_filename, "r") as f:
        dset_names = list(f.keys())
        locations = fill_missing(f["tracks"][:].T)
        node_names = [n.decode() for n in f["node_names"][:]]

    print("===filename===")
    print(h5_filename)
    print()

    print("===HDF5 datasets===")
    print(dset_names)
    print()

    print("===locations data shape===")
    print(locations.shape)
    print()

    print("===nodes===")
    for i, name in enumerate(node_names):
        print(f"{i}: {name}")
    print()

    frame_count, node_count, _, instance_count = locations.shape

    print("frame count:", frame_count)
    print("node count:", node_count)
    print("instance count:", instance_count)

def visualize_tracks(h5_filepath, track_map_out):

    with h5py.File(h5_filepath, "r") as f:
        dset_names = list(f.keys())
        locations = fill_missing(f["tracks"][:].T)
        node_names = [n.decode() for n in f["node_names"][:]]

    BODY_INDEX = 4
    body_loc = locations[:, BODY_INDEX, :, :]

    plt.figure(figsize=(7,7))
    plt.plot(body_loc[:,0,0],body_loc[:,1,0], 'y',label='mouse-0')
    plt.legend()

    plt.xlim(0,1024)
    plt.xticks([])

    plt.ylim(0,1024)
    plt.yticks([])
    plt.title('Body tracks')
    plt.savefig(track_map_out)

def visualize_and_export_velocity(h5_filepath, coord_vel_graphs_out, csv_out):

    with h5py.File(h5_filepath, "r") as f:
        dset_names = list(f.keys())
        locations = fill_missing(f["tracks"][:].T)
        node_names = [n.decode() for n in f["node_names"][:]]

    BODY_INDEX = 4
    body_loc = locations[:, BODY_INDEX, :, :]

    body_vel_mouse = smooth_diff(body_loc[:, :, 0]).tolist()
    vel_mouse_to_cm_s = [(i*(2.54/96)*(30)) for i in body_vel_mouse]

    """print(len(body_vel_mouse))
    print(max(body_loc[:, 0, 0]))
    print(min(body_loc[:, 0, 0]))
    print(max(body_loc[:, 1, 0]))
    print(min(body_loc[:, 1, 0]))"""
    

    fig = plt.figure(figsize=(15,7))
    fig.tight_layout(pad=10.0)
    
    x_coord_pix = body_loc[:, 0, 0]

    y_coord_pix = body_loc[:, 1, 0]

    x_coord_cm = [(i*(2.54/96)) for i in body_loc[:, 0, 0].tolist()]

    y_coord_cm = [(i*(2.54/96)) for i in body_loc[:, 1, 0].tolist()]

    range_min = 0
    range_max = len(body_vel_mouse)/30
    #step is 30 Hz, so 0.033 s in 1 frame, but lets do step every 10,000 frame so 10,000 * 0.033
    step = 0.03333333
    x_axis_time = np.arange(range_min, range_max, step).tolist()[:-1]
    #print(max(x_axis_time))

    ax1 = fig.add_subplot(211)
    # the format is (x,y,**kwargs)
    ax1.plot(x_axis_time, x_coord_cm, 'k', label='x')
    ax1.plot(x_axis_time, y_coord_cm, 'r', label='y')
    ax1.legend()
    #plt.xlim(0,1024)
    ax1.set_xticks([i for i in range(0,len(body_vel_mouse), 600)])
    #ax1.set_xticks(x_axis)
    #ax1.set_xticklabels(ax1.get_xmajorticklabels(), fontsize=7)
    ax1.set_title('Body X-Y Dynamics')
    ax1.set_ylabel("Coordinate (cm)")

    ax2 = fig.add_subplot(212, sharex=ax1)
    #ax2.imshow(body_vel_mouse[:,np.newaxis].T, aspect='auto', vmin=0, vmax=10)
    #plt.xlim(0,1024)
    ax2.plot(x_axis_time, vel_mouse_to_cm_s, label='Forward Speed')
    ax2.set_yticks([i for i in range(0,28, 4)])
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Speed (cm/s)")
    #ax2.legend()

    plt.savefig(coord_vel_graphs_out)
    export_to_csv(csv_out,
                  idx_time=x_axis_time,
                  idx_frame=[i for i in range(1,len(body_vel_mouse)+1)],
                  x_pix=x_coord_pix,
                  y_pix=y_coord_pix,
                  x_cm=x_coord_cm,
                  y_cm=y_coord_cm,
                  vel_f_p=body_vel_mouse,
                  vel_cm_s=vel_mouse_to_cm_s)

def export_to_csv(out_path, **kwargs):

    df = pd.DataFrame.from_dict(kwargs)

    df.to_csv(out_path, index=False)
    
def main():
    h5_filepath = r"/media/rory/Padlock_DT/Velocity/BLA-Insc-8/RDT_D1/BLA-Insc-8_RDT_D12021-10-21T11_18_16.avi.220201_145905.predictions.h5"
    track_map_out = r"/media/rory/Padlock_DT/Velocity/BLA-Insc-8/RDT_D1/track_map.png"
    coord_vel_graphs_out = r"/media/rory/Padlock_DT/Velocity/BLA-Insc-8/RDT_D1/coords_vel_graphs.png"
    csv_out = r"/media/rory/Padlock_DT/Velocity/BLA-Insc-8/RDT_D1/sleap_data.csv"

    meta_data(h5_filepath)
    visualize_tracks(h5_filepath,track_map_out)
    visualize_and_export_velocity(h5_filepath,coord_vel_graphs_out,csv_out)


if __name__ == "__main__":
    main()