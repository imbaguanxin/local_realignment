import cv2
import numpy as np 
import os 


STRIDE_RATE = 0.032


def cut_image(img, stride_w=None, stride_h=None):
    if stride_w is None:
        stride_w = max(int(img.shape[0] * STRIDE_RATE), 1)
    if stride_h is None:
        stride_h = max(int(img.shape[0] * STRIDE_RATE), 1)
    pattern_img = img[stride_h:img.shape[0] - stride_h, stride_w: img.shape[1] - stride_w]
    return pattern_img


def realignment(img_list, method, stride_w, stride_h)：
    img_width = img_list[0].shape[0] - 2 * stride_w
    img_height = img_list[0].shape[1] - 2 * stride_h
    off_set = np.array([[0, 0]])

    true_img_list = [cut_image(img_list[0], stride=(stride_w, stride_h))]
    # shift_x, shift_y: width and height
    shift_back_vec_list = [(0, 0)]

    for i in range(len(img_list) - 1):
        img_base = img_list[i]
        img_pattern = cut_image(img_list[i + 1], stride_w=stride_w, stride_h=stride_h)

        # pattern matching
        res = cv2.matchTemplate(img_base, img_pattern, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        off_set = off_set + top_left - np.array([[stride_w, stride_h]])

        # calculate img
        true_loc = (-off_set[0][1] + stride_h, -off_set[0][0] + stride_w)
        true_img = img_list[i + 1][true_loc[0]: true_loc[0] + img_height, true_loc[1]: true_loc[1] + img_width]
        true_img_list.append(true_img)
        shift_back_vec_list.append((-off_set[0][0], -off_set[0][1]))
    return true_img_list, shift_back_vec_list