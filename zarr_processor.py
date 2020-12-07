from PIL import Image
import daisy
import numpy as np
import os

class ZarrProcessor:

    def __init__(self, raw_file, raw_ds):
        self.raw_file = raw_file
        self.raw_ds = raw_ds
        self.cutout_ds = daisy.open_ds(raw_file, raw_ds)
    
    def get_image(self, coord_begin, coord_end):
        voxel_size = self.cutout_ds.voxel_size
        coord_begin = daisy.Coordinate(np.flip(np.array(coord_begin))) * voxel_size
        coord_end = daisy.Coordinate(np.flip(np.array(coord_end))) * voxel_size

        roi_offset = coord_begin
        roi_shape = coord_end - coord_begin
        roi = daisy.Roi(roi_offset, roi_shape)

        ndarray = self.cutout_ds.to_ndarray(roi=roi)
        return ndarray
    
    def write_to_tiff(self, img, fpath):
        tile = Image.fromarray(img)
        tile.save(fpath, quality=95)

    def get_image_list(self, coord_begin, coord_end):
        z_range = range(coord_begin[2], coord_end[2])
        left_top = [coord_begin[0], coord_begin[1], coord_begin[2]]
        right_bottom = [coord_end[0], coord_end[1], coord_begin[2]]
        result = []
        for z in z_range:
            left_top[2] = z
            right_bottom[2] = z + 1
            img = self.get_image(left_top, right_bottom)[0]
            result.append(img)
        return result


def test1(): 
    raw_file = '/n/groups/htem/temcagt/datasets/cb2/zarr_volume/cb2_v3.zarr'
    raw_ds = "volumes/raw_mipmap/s2_rechunked"
    zp = ZarrProcessor(raw_file, raw_ds)
    img_list = zp.get_image_list(
        coord_begin=[10000, 10000, 100],
        coord_end=[11000, 10800, 120]
    )
    for i, img in enumerate(img_list):
        fpath = os.path.join('/n/groups/htem/Segmentation/xg76/local_realignment/1204', str(i) + '.tiff')
        zp.write_to_tiff(img, fpath)


if __name__ == "__main__":
    test1()    


        

