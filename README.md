# local realignment

```shell
activatedaisy
python realignment.py config.json
```

sample onfig:
```json
{
    "output_path": "/n/groups/htem/Segmentation/xg76/local_realignment/result",
    "log_path": "/n/groups/htem/Segmentation/xg76/local_realignment/result/log",
    "zarr_info": {
        "read_file": "/n/groups/htem/temcagt/datasets/cb2/zarr_volume/cb2_v3.zarr",
        "read_ds": "volumes/raw_mipmap/s2_rechunked",
        "stride_rate": 0.032
    },
    "realign_info":{
        "coord_begin": [
            10000,
            11000,
            100
        ],
        "coord_end": [
            11000,
            12000,
            120
        ],
        "interval": 1,
        "img": true,
        "description_img": true
    }
}
```
- output_path: the results goes

- log_path: optional, default is {output_path}/log

- stride_rate: how large the stride image (percentage length expand on each side)

- coord_begin: the coord of left upper top corner

- coord_end: the coord of right lower bottom corner

- interval: optional, sample every {interval} to get a picture, default 1

- img: optional, whether to output tiff imgs, default false

- description_img: optional, whether output description img, default false (for making demos)

Result: {output_path}/xxxx.result.csv

shift of each image according to the top image.