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
import matplotlib.pyplot as pyplot
import random

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


def old_demo():
    # DONT RUN THIS DEMO
    def preprocess_img():
        path = 'D:/repos/local_realignment/img'
        img_name_list = [str(n) + '.png' for n in range(596, 601)]
        cut_img = []
        for img_name in img_name_list:
            img = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)
            shift_x = random.randint(0, 32)
            shift_y = random.randint(0, 32)
            img_start_x = 400 + shift_x
            img_start_y = 250 + shift_y
            img = img[img_start_x:img_start_x + 1064, img_start_y:img_start_y + 1064]
            pattern = img[32:1032, 32:1032]
            cut_img.append({'img': img, 'pattern': pattern, 'x': shift_x, 'y': shift_y})
            # plt.imshow(img, cmap='gray')
            # plt.axis('off')
            # plt.show()
        for idx, img in enumerate(cut_img):
            img = Image.fromarray(img['img'])
            fn = os.path.join(path, "img" + str(idx) + ".png")
            img.save(fn)
        return cut_img


    def paste_img(img_array, background, off_x, off_y):
        bg = background.copy()
        img = Image.fromarray(img_array)
        bg.paste(img, (off_x, off_y))
        return bg


    def load_img():
        test_img_path = 'D:/repos/local_realignment/test_img'
        fn_list = os.listdir(test_img_path)
        test_image_list = [cv2.imread(os.path.join(test_img_path, fn), cv2.IMREAD_GRAYSCALE) for fn in fn_list]
        cut_img = []
        for img in test_image_list:
            pattern = img[32:1032, 32:1032]
            cut_img.append({'img': img, 'pattern': pattern})
        return cut_img


    def main():
        num = 0
        img_result_list = []
        fig_list = []
        background = Image.new('RGB', (1400, 1400), (0, 0, 0))
        off_set = np.array([[168, 168]])
        path = 'D:/repos/local_realignment'
        # cut_img_list = preprocess_img()
        cut_img_list = load_img()

        for img_dict in cut_img_list:
            img = paste_img(img_dict['img'], background, 168, 168)
            img_result_list.append(img)
            fig = plt.figure()
            # plt.imshow(img, cmap='gray')
            # plt.axis('off')
            # plt.title('Original Layout')
            # plt.show()
            # plt.savefig(os.path.join(path, str(num) + '.png'))

            num += 1

        img_1 = paste_img(cut_img_list[0]['img'], background, 168, 168)
        img_result_list.append(img_1)
        plt.imshow(img_1, cmap='gray')
        plt.axis('off')
        plt.title('Changed Layout')
        # plt.show()
        # plt.savefig(os.path.join(path, str(num) + '.png'))
        num += 1

        for i in range(len(cut_img_list) - 1):
            img_base = cut_img_list[i]['img']
            img_pattern = cut_img_list[i + 1]['pattern']
            res = cv2.matchTemplate(img_base, img_pattern, METHOD)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            top_left = max_loc

            off_set = off_set + top_left - 32
            print(off_set - 168)
            img = paste_img(cut_img_list[i + 1]['img'], background, off_set[0][0], off_set[0][1])
            img_result_list.append(img)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.title('Changed Layout')
            # plt.show()
            # plt.savefig(os.path.join(path, str(num) + '.png'))
            num += 1


if __name__ == "__main__":
    test_write_to_zarr()
