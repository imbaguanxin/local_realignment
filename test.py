import os
from PIL import Image
import re
import shutil
# from zarr_tiff import down_sampling_img
from PIL import Image, ImageOps
import numpy as np
import shutil
import daisy
import cv2


def make_gif():
    fig_dir = '/n/groups/htem/users/xg76/local_realignment/img_result'
    fig_list = []
    print(fig_list)
    figs = list(map(lambda fn: Image.open(
        os.path.join(fig_dir, fig_list)), fig_list))
    result_path = '/n/groups/htem/users/xg76/local_realignment'
    figs[0].save(os.path.join(result_path, 'out.gif'),
                 save_all=True,
                 append_images=figs[1:],
                 optimize=False,
                 duration=500)


def test():
    f_name = '/n/groups/htem/Segmentation/xg76/local_realignment/test.zarr'
    data_set = 'volumes/raw'
    roi = daisy.Roi((0, 0, 0), (5000, 5000, 5000))
    voxel_size = np.array([40, 4, 4])
    zarr_out = daisy.prepare_ds(
        f_name,
        data_set,
        roi,
        voxel_size,
        np.uint8
    )

    write_roi = daisy.Roi((0, 0, 0), (40, 3200, 4000))
    i = cv2.imread(
        '/n/groups/htem/Segmentation/xg76/local_realignment/1204/0.tiff',
        cv2.IMREAD_GRAYSCALE)
    shape = (np.array(write_roi.get_shape()) / voxel_size).astype(int)
    arr = np.zeros(shape, dtype=np.uint8)
    arr[0, :, :] = i
    zarr_out[write_roi] = arr


if __name__ == "__main__":
    test()
