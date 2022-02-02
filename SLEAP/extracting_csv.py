import os
import h5py
import numpy as np
import pandas as pd
import sys
#export one .slp file to .h5 to txt to csv
def slp_to_h5():

    input_path = r"/media/rory/Padlock_DT/SLEAP/predictions/BLA-Insc-8_RDT_D12021-10-21T11_18_16.avi.220201_145905.predictions.slp"

    output_path = r"/media/rory/Padlock_DT/SLEAP/predictions/BLA-Insc-8_RDT_D12021-10-21T11_18_16.avi.220201_145905.predictions.h5"
    cmd = f"sleap-convert --format analysis -o {output_path} {input_path}"
    os.system(cmd)

def h5_to_csv():

    filename = "/media/rory/Padlock_DT/SLEAP/predictions/BLA-Insc-8_RDT_D12021-10-21T11_18_16.avi.220201_145905.predictions.h5"

    df = pd.read_hdf(filename)
    df.to_csv("/media/rory/Padlock_DT/SLEAP/predictions/BLA-Insc-8_RDT_D12021-10-21T11_18_16.avi.220201_145905.predictions.csv",index=False)


if __name__ == "__main__":
    #slp_to_h5()
    h5_to_csv()