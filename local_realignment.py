import cv2
import numpy as np 
import os 
from zarr_processor import ZarrProcessor


class LocalRealignment:

    def __init__(self, raw_file, raw_ds, stride_rate, method):
        super().__init__()
        self.zarr = ZarrProcessor(raw_file, raw_ds)
        self.stride_rate = stride_rate
        self.method = method

    def realign(self, coord_begin, coord_end):
        width = abs(coord_end[0] - coord_begin[0])
        height = abs(coord_end[1] - coord_begin[1])
        # print(f'{width}, {height}')
        s_w = max(1, int(width * self.stride_rate))
        s_h = max(1, int(height * self.stride_rate))
        # print(f'{s_w}, {s_h}')
        stride_coord_begin = np.array(coord_begin) - np.array([s_w, s_h, 0])
        stride_coord_end = np.array(coord_end) + np.array([s_w, s_h, 0])
        # print(f'{stride_coord_begin}, {stride_coord_end}')
        raw_img_list = self.zarr.get_image_list(stride_coord_begin, stride_coord_end)
        # for i, img in enumerate(raw_img_list):
        #     self.zarr.write_to_tiff(img, f'/n/groups/htem/Segmentation/xg76/local_realignment/1204_lr/rawimg_{i}.tiff')
        pattern_list = list(map(self.__cut_image, raw_img_list))
        # for i, img in enumerate(pattern_list):
        #     self.zarr.write_to_tiff(img, f'/n/groups/htem/Segmentation/xg76/local_realignment/1204_lr/pattern_{i}.tiff')

        off_set = np.array([[0, 0]])
        # shift_x, shift_y: width and height
        shift_back_vec_list = [(0, 0)]

        true_img_list = [pattern_list[0]]
        for i in range(len(raw_img_list) - 1):
            img_base = raw_img_list[i]
            img_pattern = pattern_list[i]

            # pattern matching
            res = cv2.matchTemplate(img_base, img_pattern, self.method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if self.method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            off_set = off_set + top_left - np.array([[s_w, s_h]])

            # calculate img
            true_loc = (-off_set[0][1] + s_h, -off_set[0][0] + s_w)
            print(true_loc)
            true_img = raw_img_list[i + 1][true_loc[0]: true_loc[0] + height, true_loc[1]: true_loc[1] + width]
            true_img_list.append(true_img)
            shift_back_vec_list.append((-off_set[0][0], -off_set[0][1]))
        return true_img_list, shift_back_vec_list

    def __cut_image(self, img, stride_w=None, stride_h=None):
        if stride_w is None:
            stride_w = max(int(img.shape[0] * self.stride_rate), 1)
        if stride_h is None:
            stride_h = max(int(img.shape[1] * self.stride_rate), 1)
        pattern_img = img[stride_h:img.shape[0] - stride_h, stride_w: img.shape[1] - stride_w]
        return pattern_img

    
if __name__ == "__main__":
    # raw_file = '/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_gt/synapse_gt/ml0/ml0.zarr'
    raw_file = '/n/groups/htem/temcagt/datasets/cb2/zarr_volume/cb2_v3.zarr'
    raw_ds = "volumes/raw_mipmap/s2_rechunked"

    lr = LocalRealignment(
        raw_file=raw_file,
        raw_ds=raw_ds,
        stride_rate=0.032,
        method=cv2.TM_CCORR)
    img_list, shift_vec = lr.realign(
        coord_begin=[10000, 10000, 100],
        coord_end=[11000, 10800, 120])
    for i, img in enumerate(img_list):
        fpath = os.path.join('/n/groups/htem/Segmentation/xg76/local_realignment/1204_lr', str(i) + '.tiff')
        print(img.shape)
        lr.zarr.write_to_tiff(img, fpath)

# STRIDE_RATE = 0.032


# def cut_image(img, stride_w=None, stride_h=None):
#     if stride_w is None:
#         stride_w = max(int(img.shape[0] * STRIDE_RATE), 1)
#     if stride_h is None:
#         stride_h = max(int(img.shape[0] * STRIDE_RATE), 1)
#     pattern_img = img[stride_h:img.shape[0] - stride_h, stride_w: img.shape[1] - stride_w]
#     return pattern_img


# def realignment(img_list, method, stride_w, stride_h):
#     img_width = img_list[0].shape[0] - 2 * stride_w
#     img_height = img_list[0].shape[1] - 2 * stride_h
#     off_set = np.array([[0, 0]])

#     true_img_list = [cut_image(img_list[0], stride=(stride_w, stride_h))]
#     # shift_x, shift_y: width and height
#     shift_back_vec_list = [(0, 0)]

#     for i in range(len(img_list) - 1):
#         img_base = img_list[i]
#         img_pattern = cut_image(img_list[i + 1], stride_w=stride_w, stride_h=stride_h)

#         # pattern matching
#         res = cv2.matchTemplate(img_base, img_pattern, method)
#         min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#         if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
#             top_left = min_loc
#         else:
#             top_left = max_loc
#         off_set = off_set + top_left - np.array([[stride_w, stride_h]])

#         # calculate img
#         true_loc = (-off_set[0][1] + stride_h, -off_set[0][0] + stride_w)
#         true_img = img_list[i + 1][true_loc[0]: true_loc[0] + img_height, true_loc[1]: true_loc[1] + img_width]
#         true_img_list.append(true_img)
#         shift_back_vec_list.append((-off_set[0][0], -off_set[0][1]))
#     return true_img_list, shift_back_vec_list