import nibabel as nib
import math

import os
import glob
import shutil
from tqdm import tqdm
import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage.exposure import rescale_intensity

P = [252, 233, 79, 114, 159, 207, 239, 41, 41, 173, 127, 168, 138, 226, 52,
     233, 185, 110, 252, 175, 62, 211, 215, 207, 196, 160, 0, 32, 74, 135, 164, 0, 0,
     92, 53, 102, 78, 154, 6, 143, 89, 2, 206, 92, 0, 136, 138, 133, 237, 212, 0, 52,
     101, 164, 204, 0, 0, 117, 80, 123, 115, 210, 22, 193, 125, 17, 245, 121, 0, 186,
     189, 182, 85, 87, 83, 46, 52, 54, 238, 238, 236, 0, 0, 10, 252, 233, 89, 114, 159,
     217, 239, 41, 51, 173, 127, 178, 138, 226, 62, 233, 185, 120, 252, 175, 72, 211, 215,
     217, 196, 160, 10, 32, 74, 145, 164, 0, 10, 92, 53, 112, 78, 154, 16, 143, 89, 12,
     206, 92, 10, 136, 138, 143, 237, 212, 10, 52, 101, 174, 204, 0, 10, 117, 80, 133, 115,
     210, 32, 193, 125, 27, 245, 121, 10, 186, 189, 192, 85, 87, 93, 46, 52, 64, 238, 238, 246]

P = P * math.floor(255 * 3 / len(P))
l = int(255 - len(P) / 3)
P = P + P[3:(l + 1) * 3]
P = [0, 0, 0] + P


def read_indexed_png(fname):
    im = Image.open(fname)
    palette = im.getpalette()
    im = np.array(im)
    return im, palette


def save_indexed_png(fname, label_map, palette=P):
    if label_map.max() > 255:
        label_map = np.remainder(label_map, 255)
    label_map = np.squeeze(label_map.astype(np.uint8))
    im = Image.fromarray(label_map, 'P')
    im.putpalette(palette)
    im.save(fname, 'PNG')


def save_indexed_tif(fname, label_map, palette=P):
    raw_map = label_map.copy()
    if label_map.max() > 255:
        label_map = np.remainder(label_map, 255)
    label_map = np.squeeze(label_map.astype(np.uint8))
    label_map[np.logical_and(raw_map, ~label_map)] = 255
    im = Image.fromarray(label_map, 'P')
    im.putpalette(palette)
    im.save(fname)


def check_folder(file_folder, overwrite=False):
    if "." in file_folder:
        file_folder = os.path.dirname(file_folder)
    if os.path.isdir(file_folder) and overwrite:
        shutil.rmtree(file_folder)
    elif not os.path.isdir(file_folder):
        os.makedirs(file_folder)


def nib_save(file_name, data, overwrite=False):
    check_folder(file_name, overwrite)
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, file_name)


def nib_load(file_name):
    assert os.path.isfile(file_name), "File {} not exist".format(file_name)

    return nib.load(file_name).get_fdata()


if __name__ == "__main__":
    dst_folder = r'D:\cell_shape_quantification\DATA\figure\3DEmbyro\tif'
    embryo_num_string = f'{4:02}'
    embryo_path_tmp = r'../DATA/SegmentCellUnified04-20/Sample' + embryo_num_string + 'LabelUnified'
    seg_files_path=[]
    for file_name in os.listdir(embryo_path_tmp):
        if os.path.isfile(os.path.join(embryo_path_tmp, file_name)):
            # print(path_tmp)
            seg_files_path.append(file_name)

    # seg_files = sorted(glob.glob(os.path.join(src_folder, "*.nii.gz")))
    print(seg_files_path)
    seg_files_path = sorted(seg_files_path)
    print(seg_files_path)
    label_file_path = os.path.join(os.path.dirname(dst_folder), "label.txt")
    print(label_file_path)

    for idx, seg_file in enumerate(tqdm(seg_files_path, desc="Svaing to {}".format(dst_folder))):
        # deal with each time points
        base_name = seg_file.split(".")[0]
        seg_embryo_path=os.path.join(embryo_path_tmp, seg_file)
        seg0 = nib_load(seg_embryo_path)
        # print(seg0)
        seg = seg0 % 255
        # print(seg)
        reduce_mask = np.logical_and(seg0 != 0, seg == 0)
        seg[reduce_mask] = 255
        seg = seg.astype(np.uint8)
        origin_shape = seg.shape
        out_size = [int(x / 1.5) for x in origin_shape]

        seg = resize(image=seg, output_shape=out_size, preserve_range=True, order=0,anti_aliasing=False).astype(np.uint8)

        # seg = seg[..., :(out_size[-1] // 2)]
        tif_imgs = []
        num_slices = seg.shape[-1]
        for i_slice in range(num_slices):
            tif_img = Image.fromarray(seg[..., i_slice], mode="P")
            tif_img.putpalette(P)
            tif_imgs.append(tif_img)
        save_file = os.path.join(dst_folder, "_".join([base_name, "render.tif"]))
        if os.path.isfile(save_file):
            os.remove(save_file)

        label_num = np.unique(seg).tolist()[-1]
        if idx == 0:
            with open(label_file_path, "w") as f:
                f.write("{}\n".format(label_num))
        else:
            with open(label_file_path, "a") as f:
                f.write("{}\n".format(label_num))

        tif_imgs[0].save(save_file, save_all=True, append_images=tif_imgs[1:])
