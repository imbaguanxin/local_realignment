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
    if coord_begin != None and coord_end != None:
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

####################################################################


def main():

    raw_file = '/n/groups/htem/temcagt/datasets/cb2/zarr_volume/cb2_v3.zarr'
    raw_ds = "volumes/raw_mipmap/s2_rechunked"
    output = '/n/groups/htem/temcagt/datasets/cb2/figure_section_tiffs/201004_data/small_roi_test'
    coord_begin = [0, 0, 0]
    coord_end= [62464, 57600, 1216]

    cutout_ds = daisy.open_ds(raw_file, raw_ds)
    roi = calc_roi(cutout_ds.voxel_size, coord_begin, coord_end)
    print(roi)

    # for i in range(coord_end[2] - coord_begin[2]):
    #     if i % 10 == 0:


    # raw_img_list = get_ndarray_img_from_zarr(raw_file, raw_ds, coord_begin, coord_end)

    # os.makedirs(output, exist_ok=True)

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
    test_write_tiff()
    # main()

    # print(script_name)
