import os
from PIL import Image

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
