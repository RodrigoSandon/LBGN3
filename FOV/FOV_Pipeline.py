import isx
import numpy as np
import pandas as pd

# project movie
"""input_movie = "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/motion_corrected.isxd"
output_image = "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/max_projection.isxd"

isx.project_movie(input_movie, output_image, stat_type = "max")"""
# cant export isxd to png

# filter cell set for only accepted cells (needs to run separately on fov_test)
"""input_movie = isx.Movie.read("/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/motion_corrected.isxd")
cell_set = isx.CellSet.read("/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/cnmfe_cellset.isxd")
cell_set_out = isx.CellSet.write("/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/cnmfe_cellset_accepts.isxd", input_movie.timing, cell_set.spacing)
num_cells = cell_set.num_cells

idx = 0
for i in range(num_cells):
    cell_status = cell_set.get_cell_status(i)
    if str(cell_status) == "accepted":
        #print(cell_status)
        image = cell_set.get_cell_image_data(i)
        trace =  cell_set.get_cell_trace_data(i)
        idx_for_name = idx + 1
        if idx_for_name <= 9:
            name = f"C0{idx_for_name}"
        else:
            name = f"C{idx_for_name}"
        cell_set_out.set_cell_data(idx,image,trace,name)
        idx += 1
"""
# get centroids
cell_set = isx.CellSet.read("/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/cnmfe_cellset_accepts.isxd")
num_cells = cell_set.num_cells

# load cell identities of specific subevent
subevent_path = "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/BetweenCellAlignmentData/Shock Ocurred_Choice Time (s)/True/concat_cells_z_fullwindow_id_auc_bonf0.05.csv"
df_identities = pd.read_csv(subevent_path)

# hashmap of centroid coordinates : cell identity
id_coord = {"-":[],"+":[],"Neutral":[]}

idx = 0
for i in range(num_cells):

    cell_image = cell_set.get_cell_image_data(i).astype(np.float64)
    num_pixels = cell_set.spacing.num_pixels
    cell_name = cell_set.get_cell_name(i)
    id = list(df_identities[cell_name])[0] # there's only 1 value in the column

    X, Y = np.meshgrid(np.arange(0.5, num_pixels[1]),np.arange(0.5, num_pixels[0]))
    centroid = np.stack((X, Y)).reshape(2, -1).dot(cell_image.ravel()) / cell_image.sum()

    id_coord[id].append(centroid.tolist())

#print(id_coord)

# plot centroids as circles
import cv2
import pyautogui
# load projection.isxd onto ISX GUI, export as tiff, take ss, load, resize based on tiff
tiff_dims = (314, 195)

#transparent background
width, height = 314, 195  # set same size as ss taken
channels = 3
canvas = np.zeros((height,width,channels), dtype="uint8")
CIRCLE_RADIUS = 6
CIRCLE_THICKNESS = 2
COLOR_POS = (0,0,255)
COLOR_NEG = (255,0,0)
COLOR_NEU = (0,255,0)

for identity in id_coord:
    centroid_list = id_coord[identity]

    for c in centroid_list:
        x_y = (int(c[0]), int(c[1]))
        #print(x_y)
        COLOR = None
        if identity == "+":
            COLOR = COLOR_POS
        elif identity == "-":
            COLOR = COLOR_NEG
        else:
            COLOR = COLOR_NEU
        cv2.circle(canvas, x_y, CIRCLE_RADIUS, COLOR, CIRCLE_THICKNESS)

#may need to rotate, make other adjustments to image
#canvas = cv2.rotate(canvas, cv2.ROTATE_180)
cv2.imwrite("/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/shock_true_map.png", canvas)

#overlay images
labels = cv2.imread("/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/shock_true_map.png")
print(labels.shape)
#resize label based on labels img dimensions
projection = cv2.imread("/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/projection_ss.png")
projection = cv2.resize(projection, tiff_dims, interpolation=cv2.INTER_AREA)
print(projection.shape)

# alpha closer to 1.0 -> more opaque
# closer to 0.0 -> more transparent

"""labels[labels[:, :, 1:].all(axis=-1)] = 0
projection[projection[:, :, 1:].all(axis=-1)] = 0"""

result = cv2.addWeighted(labels, 0.4, projection, 0.7, 0)
#img_arr = np.hstack((labels, projection))

#cv2.imshow("Input images", img_arr)
overlay_path = "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/overlay_shock_T.png"
cv2.imwrite(overlay_path, result)
    