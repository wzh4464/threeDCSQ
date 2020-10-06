import nibabel as nib
import os
from nibabel.viewers import OrthoSlicer3D


def load_nitf2_img(path):
    return nib.load(path)


def show_nitf2_img(path):
    img = load_nitf2_img(path)
    OrthoSlicer3D(img.dataobj).show()
    return img


def deal_all(this_dir):
    for (root, dirs, files) in os.walk(this_dir):
        # print(files)
        for file in files:
            print(root, file)
            this_file_path = os.path.join(root, file)
            this_img = nib.load(this_file_path)
            print(this_img.shape)
