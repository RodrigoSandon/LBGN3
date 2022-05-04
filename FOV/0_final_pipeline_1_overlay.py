import isx
import numpy as np
import pandas as pd

movie_path = "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-15/RDT D1/Session-20220121-100647_BLA-Insc-15_RDT_D1/motion_corrected.isxd"
output_image = "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#5/BLA-Insc-15/RDT D1/Session-20220121-100647_BLA-Insc-15_RDT_D1/max_projection.isxd"

isx.project_movie(movie_path, output_image, stat_type = "max")

