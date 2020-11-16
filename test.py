import os
from PIL import Image
import re
import shutil
from zarr_tiff import down_sampling_img
from PIL import Image, ImageOps
import numpy as np
import shutil


def make_gif():
    fig_dir = '/n/groups/htem/users/xg76/local_realignment/img_result'
    fig_list = []
    print(fig_list)
    figs = list(map(lambda fn: Image.open(os.path.join(fig_dir, fig_list)), fig_list))
    result_path = '/n/groups/htem/users/xg76/local_realignment'
    figs[0].save(os.path.join(result_path, 'out.gif'),
                 save_all=True,
                 append_images=figs[1:],
                 optimize=False,
                 duration=500)

def cp_img():
    test_img_dir = '/n/groups/htem/users/xg76/local_realignment/test_img'
    s3_dir = os.listdir(test_img_dir)
    s3_dir = list(filter(lambda w: bool(re.search(re.compile('\d*_s3'), w)), s3_dir))
    os.makedirs(os.path.join(test_img_dir, 's3'), exist_ok=True)
    for s3 in s3_dir:
        tifs = os.listdir(os.path.join(test_img_dir, s3))
        for tif in tifs:
            shutil.copyfile(os.path.join(test_img_dir, s3, tif), os.path.join(test_img_dir, 's3', tif))


def check_img():
    s3_dir = os.listdir('/n/groups/htem/users/xg76/local_realignment/test_img/s3')
    s3_dir = set(map(lambda w: int(re.findall(re.compile('\d\d+'), w)[0]), s3_dir))
    all_img = set(range(70, 1220, 10))
    diff = all_img.difference(s3_dir)
    complete = diff == set()
    print(f'check img result: {complete}, difference: {diff}')


def downsample():
    test_img_dir = '/n/groups/htem/Segmentation/xg76/local_realignment/mipmap'
    s3_dir = '/n/groups/htem/Segmentation/xg76/local_realignment/mipmap/s3'
    s3_pics = os.listdir(s3_dir)
    os.makedirs(os.path.join(test_img_dir, 's3x2'), exist_ok=True)
    os.makedirs(os.path.join(test_img_dir, 's3x3'), exist_ok=True)
    os.makedirs(os.path.join(test_img_dir, 's3x4'), exist_ok=True)
    for picname in s3_pics:
        print(picname)
        img = Image.open(os.path.join(s3_dir, picname))
        img = ImageOps.grayscale(img)
        img = np.array(img)
        scale_list = [2,3,4]
        downsample_list = down_sampling_img(img, scale_list)
        for idx, img in enumerate(downsample_list):
            tile = Image.fromarray(img)
            scale = scale_list[idx]
            tile.save(os.path.join(test_img_dir, f's3x{scale}', f'{picname}_x{scale}.tiff'))


def copy_file():
    mipmap_dir = '/n/groups/htem/Segmentation/xg76/local_realignment/mipmap'
    dest_dir = '/n/groups/htem/Segmentation/xg76/local_realignment/mipmap/s3'
    mipmap_folder = os.listdir(mipmap_dir)
    for mipmaplayer in mipmap_folder:
        mipmap_files = os.listdir(os.path.join(mipmap_dir, mipmaplayer))
        for mipmap in mipmap_files:
            try:
                if '1000' in mipmaplayer:
                    shutil.copyfile(os.path.join(mipmap_dir, mipmaplayer, mipmap), os.path.join(dest_dir, mipmap))
                else:
                    shutil.copyfile(os.path.join(mipmap_dir, mipmaplayer, mipmap), os.path.join(dest_dir, mipmap + 'f'))
            except:
                print(f'{mipmap} failed')


if __name__ == "__main__":
    downsample()