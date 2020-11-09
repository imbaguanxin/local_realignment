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


def get_ndarray_img_from_zarr(raw_file=None, raw_ds=None, coord_begin=None, coord_end=None, cutout_ds=None):
    """
    Retrieve image from zarr file.
    Return list of images
    """
    if raw_file is None and raw_ds is None and cutout_ds is None:
        raise ValueError('No raw file is found')
    elif raw_file is not None and raw_ds is not None and cutout_ds is None:
        cutout_ds = daisy.open_ds(raw_file, raw_ds)
    else:
        print('Using passed in cutout_ds')
    print(f'Voxel size: {cutout_ds.voxel_size}')
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


def main_origin(section, z_start, z_end):
    interval = 10
    # S2: 3280-21435, 2840 22160, 70-1170
    # S4: 1000-14000, 1000 14000, 35-640
    # S5: 500-7000, 500 7000, 15-300
    section_info = {
        's5':{
            'raw_ds': "volumes/raw_mipmap/s5_rechunked",
            'coord_begin': [500, 500, 15],
            'coord_end': [7000, 7000, 300],
            'z_range': range(15, 300, interval),
            'scale': [8]
        },
        's4':{
            'raw_ds': "volumes/raw_mipmap/s4_rechunked",
            'coord_begin': [1000, 1000, 40],
            'coord_end': [15616, 14400, 640],
            'z_range': range(40, 640, interval),
            'scale': [16]
        },
        's2':{
            'raw_ds': "volumes/raw_mipmap/s2_rechunked",
            'coord_begin': [3280, 2840, 70],
            'coord_end': [62464, 57600, 1180],
            'z_range': range(z_start, z_end, interval),
            'scale': [64]
        },
        's3':{
            'raw_ds': "volumes/raw_mipmap/s3_rechunked",
            'coord_begin': [1500, 1400, 70],
            'coord_end': [31232, 28800, 1210],
            'z_range': range(z_start, z_end, interval),
            'scale': [16]
        }
    }
    print(f'zarr_tiff main_origin: {section}')
    s = section_info[section]
    raw_file = '/n/groups/htem/temcagt/datasets/cb2/zarr_volume/cb2_v3.zarr'
    raw_ds = s['raw_ds']
    now = datetime.now().strftime("%m%d.%H.%M.%S")
    output = f'/n/groups/htem/users/xg76/local_realignment/test_img/{now}_{section}'

    coord_begin = s['coord_begin']
    coord_end = s['coord_end']
    z_range = s['z_range']
    scale_list = s['scale']
    os.makedirs(output, exist_ok=True)

    cutout_ds = daisy.open_ds(raw_file, raw_ds)

    for z in z_range:
        coord_begin[2] = z
        coord_end[2] = coord_begin[2] + 1
        print(f'coord begin: {coord_begin}')

        raw_img_list = get_ndarray_img_from_zarr(
            coord_begin=coord_begin, 
            coord_end=coord_end,
            cutout_ds=cutout_ds)
        img = raw_img_list[0]
        # write_to_tiff(img, os.path.join(output, f'{z}_origin.tif'))
        mipmap = down_sampling_img(img, scale_list)[0]
        write_to_tiff(mipmap, os.path.join(output, f'{section}_{z}_{scale_list[0]}_mipmap.tif'))


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
    if len(sys.argv) == 4:
        sec = sys.argv[1]
        z_s = int(sys.argv[2])
        z_e = int(sys.argv[3])
        main_origin(sec, z_s, z_e)
    else:
        print('Something wrong with argument\n should be "section_num start_z end_z"')
    