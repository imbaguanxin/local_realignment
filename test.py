import os
from PIL import Image
import re
import shutil
from PIL import Image, ImageOps
import numpy as np
import shutil
import daisy
import cv2
import numpy as np 
from scipy.stats import pearsonr

CROSS_CORRELATION = 'xcorr'
NORMALIZED_CROSS_CORRELATION = 'nxorr'
CHUNKED_PEAESON_CORRELATION = 'pearsonr'


class PatternTest:

    @staticmethod
    def _normalize_img(img):
        f_img = img.flatten()
        return (f_img - np.mean(f_img)) / (np.std(f_img) * len(f_img))

    def _calc(self, img_base, img_test, method):
        if method == CROSS_CORRELATION:
            result = np.correlate(img_base.flatten(), img_test.flatten())[0]
        elif method == NORMALIZED_CROSS_CORRELATION:
            result = np.correlate(self._normalize_img(img_base), self._normalize_img(img_test))[0]
        elif method == CHUNKED_PEAESON_CORRELATION:
            result = pearsonr(img_base.flatten(), img_test.flatten())[0]
        else:
            raise ValueError('Test Method not found: {}'.format(method))
        return result
    
    def test_main(self, img_list, methods):
        result_list = []
        for i in range(len(img_list) - 1):
            result_list.append(self._calc(img_list[i], img_list[i + 1]))
        return result_list


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


def test_write_to_zarr():
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
    test_write_to_zarr()
