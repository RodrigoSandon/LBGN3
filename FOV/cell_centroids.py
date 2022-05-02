import isx
import numpy as np

cell_set = isx.CellSet.read("/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/cnmfe_cellset_accepts.isxd")
num_cells = cell_set.num_cells

idx = 0
for i in range(num_cells):

    cell_image = cell_set.get_cell_image_data(i).astype(np.float64)
    num_pixels = cell_set.spacing.num_pixels

    X, Y = np.meshgrid(np.arange(0.5, num_pixels[1]),np.arange(0.5, num_pixels[0]))
    centroid = np.stack((X, Y)).reshape(2, -1).dot(cell_image.ravel()) / cell_image.sum()

    print(centroid)