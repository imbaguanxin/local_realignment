import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from zarr_reader import ZarrReader
import logging
import sys
import json
import datetime
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd


class Realignment:

    def __init__(
        self,
        read_file,
        read_ds,
        stride_rate=0.032,):
        self.stride_rate = stride_rate
        self.zarr_reader = ZarrReader(read_file=read_file, read_ds=read_ds)

    # this function is not used
    def __cut_image(self, img, stride):
        if type(stride) is int:
            str_r = str_c = stride
        elif type(stride) is tuple:
            str_r = stride[0]
            str_c = stride[1]
        else:
            str_r = max(int(img.shape[0] * self.stride_rate), 1)
            str_c = max(int(img.shape[1] * self.stride_rate), 1)
        pattern_img = img[str_r:img.shape[0] - str_r, str_c: img.shape[1] - str_c]
        return pattern_img

    def calc_shift(
            self, 
            stride_list,
            pattern_list, 
            stride_r, 
            stride_c, 
            method=cv2.TM_CCOEFF):
        """
        stride_list: [list of np.array] input image with stride (larger)
        pattern_list: [list of np.array] input image (smaller)
        stride_w: stride in of w in pixel.
        stride_h: stride in of h in pixel.

        return: [list of tuple] shift of each picture in pixel.
        """
        # checking img length
        stride_length = len(stride_list)
        pattern_length = len(pattern_list)
        if stride_length != pattern_length:
            raise ValueError(f'stride image length {stride_length} not eqaul to pattern image length {pattern_length}')

        # shift_x, shift_y: width and height
        off_set = np.array([[0, 0]])
        off_set_list = [(0,0)]

        for i in range(len(stride_list) - 1):
            img_base = stride_list[i]
            img_pattern = pattern_list[i + 1]
            #self.__cut_image(img_list[i + 1], stride=(stride_w, stride_h))

            # pattern matching
            res = cv2.matchTemplate(img_base, img_pattern, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc

            off_set = off_set + top_left - np.array([[stride_r, stride_c]])
            off_set_list.append((off_set[0][0], off_set[0][1]))
            # calculate img
        return off_set_list
    
    def get_image_from_zarr(self, coord_begin, coord_end, interval=1):
        # get image from zarr file, getting stride and cut the pattern out
        image_row = abs(coord_begin[0] - coord_end[0])
        image_col = abs(coord_begin[1] - coord_end[1])
        stride_r = max(1, int(image_row * self.stride_rate))
        stride_c = max(1, int(image_col * self.stride_rate))

        stride_begin = [coord_begin[0] - stride_r,
                        coord_begin[1] - stride_c,
                        coord_begin[2]]
        stride_end = [coord_end[0] + stride_r,
                      coord_end[1] + stride_c,
                      coord_end[2]]
        stride_img = self.zarr_reader.get_image_list(stride_begin, stride_end, interval)
        pattern_img = [np.array(i[stride_r: image_row, stride_c: image_col]) for i in stride_img] 
        # self.zarr_reader.get_image_list(coord_begin, coord_end, interval)
        return pattern_img, stride_img, stride_r, stride_c
    
    def realign(self, 
        coord_begin, 
        coord_end,
        output_path, 
        interval=1, 
        img=True, 
        description_img=True):

        def paste_img(img, bg, off_set):
            bkgd = bg.copy()
            image = Image.fromarray(img)
            bkgd.paste(image, off_set)
            return bkgd
        
        logging.info('Retriving images ...')
        p, s, sr, sc = self.get_image_from_zarr(
            coord_begin=coord_begin,
            coord_end=coord_end,
            interval=interval
        )
        logging.info('Realigning ...')
        off_set_list = self.calc_shift(
            stride_list=p, 
            pattern_list=s, 
            stride_r=sr, 
            stride_c=sc)
        
        logging.info('Generating Output ...')
        if img:
            pth = os.path.join(output_path, 'img')
            os.makedirs(pth, exist_ok=True)
            for i, pattern in enumerate(p):
                fn = os.path.join(pth, str(i) + '.tiff')
                self.zarr_reader.write_to_tiff(pattern, fn)
        
        if description_img:
            pth = os.path.join(output_path, 'desc')
            os.makedirs(pth, exist_ok=True)
            image_row = abs(coord_begin[0] - coord_end[0])
            image_col = abs(coord_begin[1] - coord_end[1])
            background = Image.new('RGB', (int(image_row * 1.5), int(image_col * 1.5)), (0, 0, 0))
            for i, pattern in enumerate(p):
                off_set = off_set_list[i]
                off_set = (int(off_set[0] + image_row * 0.25), 
                           int(off_set[1] + image_col * 0.25))
                img_paste = paste_img(pattern, background, off_set)    
                plt.imshow(img_paste)
                plt.axis('off')
                plt.title('Shifted')
                plt.savefig(os.path.join(pth, str(i) + 'shifted.png'))
                
        return off_set_list


def main():
    if len(sys.argv) != 2:
        print('Wrong parameter format!')
        exit(1)

    file_path = sys.argv[1]
    with open(file_path, 'r') as f:
        configs = json.load(f)
    output_path = configs['output_path']
    zarr_info = configs['zarr_info']
    # create log path
    log_path = configs.get(
        'log_path',
        os.path.join(output_path, 'log'))
    os.makedirs(log_path, exist_ok=True)
    
    now = datetime.datetime.now().strftime('%m%d.%H.%M.%S')
    log_name = os.path.join(log_path, str(now)+'.log')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_name, filemode='w', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"Your config file: {configs}")
    logging.info(f"Your log file: {log_name}")

    rl = Realignment(**zarr_info)
    realign_info = configs['realign_info']
    realign_info['output_path'] = output_path
    off_set = rl.realign(**realign_info)
    
    df = pd.DataFrame(off_set)
    df.columns = ['x', 'y']
    df.to_csv(os.path.join(output_path, f'{now}.result.csv'), index=False)

if __name__ == "__main__":
    main()
    