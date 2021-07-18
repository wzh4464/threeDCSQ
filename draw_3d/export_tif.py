# save npy as tif that can be imported by ImageJ
import os
import glob
import math
import shutil
from tqdm import tqdm
import numpy as np
from PIL import Image
from skimage.transform import resize

#  =====================
#  Prerequisites
#  =====================
# ======= Index colors
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

#  ========================
#  Transform from *.npy to *.tif
#  ========================
# Set file folders

src_folder = r"C:\Users\zelinli6\OneDrive\paper_figure\PCA_matrix"
dst_folder = r"C:\Users\zelinli6\OneDrive\paper_figure\PCA_matrix"

seg_files = sorted(glob.glob(os.path.join(src_folder, "*.npy")))

for idx, seg_file in enumerate(tqdm(seg_files, desc="Svaing to {}".format(dst_folder))):

    base_name = os.path.basename(seg_file).split(".")[0]

    seg0 = np.load(seg_file)
    seg = seg0 % 255
    reduce_mask = np.logical_and(seg0 != 0, seg == 0)
    seg[reduce_mask] = 255  # Because only 255 colors are available, all cells should be numbered within [0, 255].
    seg = seg.astype(np.uint8)
    origin_shape = seg.shape
    out_size = [int(x / 1.5) for x in origin_shape]  # Reduce output size for rendering
    seg = resize(image=seg, output_shape=out_size, preserve_range=True, order=0).astype(np.uint8)

    tif_imgs = []
    num_slices = seg.shape[-1]
    for i_slice in range(num_slices):
        tif_img = Image.fromarray(seg[..., i_slice], mode="P")
        tif_img.putpalette(P)
        tif_imgs.append(tif_img)
    save_file = os.path.join(dst_folder, "_".join([base_name, "render.tif"]))
    if os.path.isfile(save_file):
        os.remove(save_file)

    # save the 1th slice image, treat others slices as appending
    tif_imgs[0].save(save_file, save_all=True, append_images=tif_imgs[1:])

#  ========================
#  Operate in Fiji « Plugins « Process « Show Color Surfaces for each image [separately]
#  ========================
