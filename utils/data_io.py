# from jf, modified by zelin

'''library for reading or writing data'''

import os
import pickle
import imageio
import shutil
import numpy as np
import pandas as pd
import nibabel as nib

#  write *.nii.gz files
def nib_save(data, file_name):
    check_folder(file_name)
    return nib.save(nib.Nifti1Image(data, np.eye(4)), file_name)

def read_new_cd(cd_file):
    df_nuc = pd.read_csv(cd_file, lineterminator="\n")
    df_nuc[["cell", "time"]] = df_nuc["Cell & Time"].str.split(":", expand=True)
    df_nuc = df_nuc.rename(columns={"X (Pixel)":"x", "Y (Pixel)":"y", "Z (Pixel)\r":"z"})
    df_nuc = df_nuc.astype({"x":float, "y":float, "z":float, "time":int})

    return df_nuc

# ===============================================
def check_folder(file_name):
    if "." in os.path.basename(file_name):
        dir_name = os.path.dirname(file_name)
    else:
        dir_name = file_name
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)