import daisy
from daisy import Coordinate, Roi
import numpy as np
from PIL import Image
import sys
import json
import os
import cv2
from tqdm import tqdm
from skimage.measure import block_reduce
from datetime import datetime


def get_ndarray_img_from_zarr(raw_file, raw_ds, coord_begin=None, coord_end=None):
    """
    Retrieve image from zarr file.
    Return list of images
    """
    cutout_ds = daisy.open_ds(raw_file, raw_ds)
    print(f'Voxel size: {cutout_ds}')
    roi = None
    if coord_begin is not None and coord_end is not None:
        voxel_size = cutout_ds.voxel_size
        coord_begin = Coordinate(np.flip(np.array(coord_begin))) * voxel_size
        coord_end = Coordinate(np.flip(np.array(coord_end))) * voxel_size

        roi_offset = coord_begin
        roi_shape = coord_end - coord_begin
        roi = Roi(roi_offset, roi_shape)
    print(f"Getting data from zarr file... ROI: {roi}")
    ndarray = cutout_ds.to_ndarray(roi=roi)
    return ndarray


def calc_roi(voxel_size, coord_begin, coord_end):
    coord_begin = Coordinate(np.flip(np.array(coord_begin))) * voxel_size
    coord_end = Coordinate(np.flip(np.array(coord_end))) * voxel_size
    roi_offset = coord_begin
    roi_shape = coord_end - coord_begin
    roi = Roi(roi_offset, roi_shape)
    return roi


def down_sampling_img(input_img, scale_list):
    """
    down sample the img to make mipmap.
    input_img: ndarray
    scale_list: int list of scale factor

    return list of images
    """
    dim = len(np.array(input_img).shape)
    # only process gray scale image or RGB image
    assert dim == 2 or dim == 3
    scale_factors = [(s, s) if dim == 2 else (s, s, 1) for s in scale_list]

    # # check img dimension
    # dimension_check = np.max([np.array(input_img.shape)[0,2] % s for s in scale_list])
    # assert dimension_check == 0

    # check type
    if issubclass(input_img.dtype.type, np.integer):
        img = np.copy(input_img) / 255.
    else:
        img = np.copy(input_img)
    
    # downsample
    result = [block_reduce(img, s, np.mean) for s in scale_factors]
    return result


def write_tiff_to_zarr(ndarray, writer):
    raise NotImplementedError()

def write_to_tiff(img, fpath):
    tile = Image.fromarray(img)
    tile.save(fpath, quality=95)

####################################################################


def main():

    raw_file = '/n/groups/htem/temcagt/datasets/cb2/zarr_volume/cb2_v3.zarr'
    raw_ds = "volumes/raw_mipmap/s2_rechunked"
    # output = '/n/groups/htem/temcagt/datasets/cb2/figure_section_tiffs/201004_data/small_roi_test'
    now = datetime.now().strftime("%m%d.%H.%M.%S")
    output = f'/n/groups/htem/users/xg76/local_realignment/test_img/{now}'
    # 3280-21435, 2840 22160, 70-1170
    # start_row = start_col = 0
    # coord_begin_row = 3280
    # coord_begin_col = 2840
    coord_begin = np.array([3280, 2840, 70])
    coord_end= [21430, 22160, 71]#[18000, 18000, 1]#[62464, 57600, 1] # 1216

    # cutout_ds = daisy.open_ds(raw_file, raw_ds)
    # adder = 3000
    # add_vec = np.array([3000, 3000, 1])
    # final_img = np.zeros((18000, 18000))
    # now_row = coord_begin_row
    # pbar = tqdm(desc='GETTING IMG', total=36)
    # for row in range(6):
    #     now_col = coord_begin_col
    #     for col in range(6):
    #         coord_begin = [now_row, now_col, 70]
    #         coord_end = [now_row + 3000, now_col + 3000, 71]
    #         print(f'now coord: {coord_begin}')
    #         roi = calc_roi(cutout_ds.voxel_size, coord_begin, coord_end)
    #         print(f'now roi: {roi}')
    #         ndarray = cutout_ds.to_ndarray(roi=roi)
    #         print(ndarray.shape)
    #         final_img[start_row: start_row+ 3000, start_col: start_col + 3000] = ndarray[0]

    #         start_col += 3000
    #         now_col += 3000
    #         pbar.update(1)
    #     now_row += 3000
    # pbar.close()

    raw_img_list = get_ndarray_img_from_zarr(raw_file, raw_ds, coord_begin, coord_end)
    # print(len(raw_img_list))
    # os.makedirs(output, exist_ok=True)
    # write_to_tiff(final_img, os.path.join(output, '0_o.tif'))
    # mipmap = down_sampling_img(final_img, [32])[0]
    # write_to_tiff(mipmap, os.path.join(output, '0_m.tif'))
    for i, img in enumerate(raw_img_list):
        write_to_tiff(img, os.path.join(output, f'{i}_o.tif'))
        mipmap = down_sampling_img(img, [32])[0]
        write_to_tiff(mipmap, os.path.join(output, f'{i}_m.tif'))


    os.makedirs(output, exist_ok=True)

    # pbar = tqdm(desc='WRITING TO TIFF', total=len(raw_img_list))
    # for idx, raw_img in enumerate(raw_img_list):
    #     if idx % 10 == 0:
    #         mipmap_img = down_sampling_img(raw_img, [32])[0]
    #         fpath = os.path.join(output, f'{idx}_32x.tiff')
    #         tile = Image.fromarray(mipmap_img)
    #         tile.save(fpath, quality=95)
    #         pbar.update(10)
    # pbar.close()




def test_write_tiff():
    now = datetime.now().strftime("%m%d.%H.%M.%S")
    # config_f = '/n/groups/htem/data/qz/200121_B2_final_gt/cube1.json'
    # script_name = os.path.basename(config_f)
    # script_name = script_name.split(".")[0]
    
    raw_file = '/n/groups/htem/data/qz/200121_B2_final.n5'
    raw_ds = "volumes/raw"
    coord_begin = [6761, 6145, 6634]
    coord_end= [7486, 6633, 6756]

    output = f'/n/groups/htem/users/xg76/local_realignment/tiffs/{now}'
    os.makedirs(output, exist_ok=True)
    pics = get_ndarray_img_from_zarr(raw_file, raw_ds, coord_begin, coord_end)
    print(len(pics))
    for idx, pic in enumerate(pics):
        fpath = f'/n/groups/htem/users/xg76/local_realignment/tiffs/{now}/{idx}.tiff'
        tile = Image.fromarray(pic)
        tile.save(fpath, quality=95)


def test_mipmapping():
    lr = '/n/groups/htem/users/xg76/local_realignment'
    pictures = os.listdir('/n/groups/htem/users/xg76/local_realignment/tiffs')
    img_list = [cv2.imread(os.path.join(lr, 'tiffs', p), -1) for p in pictures]

    scale_list = [2, 4, 6]
    for pic_idx, pic in enumerate(img_list):
        scaled_pics = down_sampling_img(pic, scale_list)
        for sc_idx, s in enumerate(scale_list):
            tile = Image.fromarray(scaled_pics[sc_idx])
            tile.save(os.path.join(lr, 'tiff_mipmap', f'{pic_idx}_{s}.tiff'))


if __name__ == "__main__":
    # print(cv2.__version__)
    # test_write_tiff()
    main()

    # print(script_name)
