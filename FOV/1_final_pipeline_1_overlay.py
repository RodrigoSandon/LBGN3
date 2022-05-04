import isx
import numpy as np
import pandas as pd
import pyautogui

############################################################################################################
################## PRIOR MANUAL STEPS:load max_projection.isxd onto ISX GUI, export as tiff ######################
############################################################################################################


def filter_cellset(movie_path, cell_set_path, filtered_cell_set_path):
    dff_movie = isx.Movie.read(movie_path)

    cell_set = isx.CellSet.read(cell_set_path)
    cell_set_out = isx.CellSet.write(filtered_cell_set_path, dff_movie.timing, cell_set.spacing)
    num_cells = cell_set.num_cells

    idx = 0
    for i in range(num_cells):
        cell_status = cell_set.get_cell_status(i)
        if str(cell_status) == "accepted":
            print(cell_status)
            image = cell_set.get_cell_image_data(i)
            trace =  cell_set.get_cell_trace_data(i)
            idx_for_name = idx + 1
            if idx_for_name <= 9:
                name = f"C0{idx_for_name}"
            else:
                name = f"C{idx_for_name}"
            cell_set_out.set_cell_data(idx,image,trace,name)
            idx += 1

movie_path = "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-15/RDT D1/Session-20220121-100647_BLA-Insc-15_RDT_D1/motion_corrected.isxd"
cell_set_path = "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-15/RDT D1/Session-20220121-100647_BLA-Insc-15_RDT_D1/cnmfe_cellset.isxd"
filtered_cell_set_path = "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-15/RDT D1/Session-20220121-100647_BLA-Insc-15_RDT_D1/cnmfe_cellset_accepts.isxd"
output_image = "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-15/RDT D1/Session-20220121-100647_BLA-Insc-15_RDT_D1/max_projection.isxd"
tiff_path = "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-15/RDT D1/Session-20220121-100647_BLA-Insc-15_RDT_D1/max_projection.tiff"
subevent_path = "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-15/RDT D1/BetweenCellAlignmentData/Shock Ocurred_Choice Time (s)/True/concat_cells_z_fullwindow_id_auc_bonf0.05.csv"
label_map_path = "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-15/RDT D1/Session-20220121-100647_BLA-Insc-15_RDT_D1/shock_true_map.png"
auto_ss = "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-15/RDT D1/Session-20220121-100647_BLA-Insc-15_RDT_D1/example_ss.png"
overlay_path = "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-15/RDT D1/Session-20220121-100647_BLA-Insc-15_RDT_D1/overlay_shock_T.png"

filter_cellset(movie_path, cell_set_path, filtered_cell_set_path)

# get centroids
cell_set = isx.CellSet.read(filtered_cell_set_path)
num_cells = cell_set.num_cells

# load cell identities of specific subevent
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

tiff_dims_list = list(cv2.imread(tiff_path).shape)
width = tiff_dims_list[1]
height = tiff_dims_list[0]
tiff_dims = (width, height)

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

cv2.imwrite(label_map_path, canvas)

#overlay images
labels = cv2.imread(label_map_path)
print(labels.shape)
#resize label based on labels img dimensions

left = 2450
top = 280
width = 1100
height = 730

im1 = pyautogui.screenshot(auto_ss, region=(left,top, width, height))
projection = cv2.imread(auto_ss)
projection = cv2.resize(projection, tiff_dims, interpolation=cv2.INTER_AREA)
print(projection.shape)

# alpha closer to 1.0 -> more opaque
# closer to 0.0 -> more transparent


result = cv2.addWeighted(labels, 0.4, projection, 0.7, 0)

cv2.imwrite(overlay_path, result)

