# OSOD-III 


## Overview
This is an official repository for **"Rectifying Open-Set Object Detection: Proper Evaluation and a Taxonomy"**.

We provide access to three datasets employed in our experiments;  
i.e., [Open Images v6](https://storage.googleapis.com/openimages/web/download_v6.html), [Caltech-UCSD Birds-200-2011 (CUB200)](https://www.vision.caltech.edu/datasets/cub_200_2011/), and [Mapillary Traffic Sign Dataset (MTSD)](https://www.mapillary.com/dataset/trafficsign).


## Datasets

#### Download Images  
We reuse existing datasets for the images without any modifications.  
To download the images, please access to each original resource and follow its instruction;
- [Open Images v6](https://storage.googleapis.com/openimages/web/download_v6.html)
- [Caltech-UCSD Birds-200-2011 (CUB200)](https://www.vision.caltech.edu/datasets/cub_200_2011/)
- [Mapillary Traffic Sign Dataset (MTSD)](https://www.mapillary.com/dataset/trafficsign)

#### Donload Annotations  
We redesinged existing annotations for the new OSOD-III scenario.  
For downloading our annotation files, please access [this link](https://www.dropbox.com/scl/fo/bb77kl1qdfynj0vc84uxi/AFVkV5pPxBGAfWp2zrSUSoI?rlkey=0d8qo4yx5m3ou8tnzv0ycbigs&st=gwggkr1a&dl=0).

**Note that each dataset must be used in compliance with the original license.**
In particular, MTSD is provided under the Creative Commons Attribution NonCommercial Share Alike (CC BY-NC-SA) license.
Please check the license terms carefully for both the images and the annotations before using the dataset.


## How to Use
In our datasets, each dataset is separeted into some **splits** according to its known/unknown classes. For each split, we have a category list ```category_X.txt``` and corresponding annotation file ```X_train.json```.

Please see ```category_X.txt``` to check the list of known/unknown categories.  
For example, ```category_t1.txt``` for CUB200 contains 50 category names of split1 as follows;
```
Black_footed_Albatross
Laysan_Albatross
Least_Auklet
Red_winged_Blackbird
...
```

Our annotation files follow [MSCOCO](https://cocodataset.org/#home)'s format.  
Thus, we can use *pycocotools* to load these annotation files and as follows;
```
from pycocotools.coco import COCO
cub200 = COCO('path/to/annotaion/t1_train.json') # this instance can be used as dataset-api
```
Please install pycocotools in advance using `pip install pycocotools` or `conda install -c conda-forge pycocotools`.

---

## Evaluation Code
We also provide our evaluation code.

### Installation
We use a repository of [OpenDet2](https://github.com/csuhan/opendet2), which is based on [Detectron2-v0.5](https://github.com/facebookresearch/detectron2/tree/v0.5).  

- Setup conda environment
<pre>
env_name=osod3
conda create -n ${env_name} python=3.8 -y
conda activate ${env_name}

# install pytorch
conda install pytorch=1.8.2 cudatoolkit=11.1 -c pytorch-lts -c nvidia
conda install pytorch=0.9.2 -c pytorch-lts -c nvidia

# install detectron2-v0.5
pip install detectron2==0.5 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
pip install timm==0.9.2

# clone this repository and build
git clone https://github.com/rsCPSyEu/OSOD-III.git
cd OSOD-III
pip install -v -e .

# [optional] if the build process does not work, change the version of setuptools may help you.
# conda install -c conda-forge setuptools=42
</pre>

- Setup dataset links
    - Open Images
        - Download the original images from [Open Images v6](https://storage.googleapis.com/openimages/web/download_v6.html).
        - Place the dataset as follows:
        <pre>
        ── datasets
            └── OpenImages
                ├── train
                │   ├── xxx.png
                │   └── ...
                ├── validation
                ├── test
                └── annotations
                    ├── animal
                    │   ├── tx_train.json
                    │   └── ...
                    └── vehicle
                        └── ...
        </pre>

    - CUB200
        - Download the original images from [Caltech-UCSD Birds-200-2011 (CUB200)](https://www.vision.caltech.edu/datasets/cub_200_2011/).
        - Place the dataset as follows:
        <pre>
        ── datasets
            └── CUB_200_2011
                ├── images
                │   ├── 001.Black_footed_Albatross
                │   ├── 002. ...
                │   └── ...
                └── random_separation
                    ├── tx_train.json
                    └── ...
        </pre>

    - MTSD
        - Download the original images from [Mapillary Traffic Sign Dataset (MTSD)](https://www.mapillary.com/dataset/trafficsign).
        - Place the dataset as follows:
        <pre>
        ── datasets
            └── Mapillary_Traffic_Sign
                ├── images
                │   ├── xxx.jpg
                │   └── ...
                └── spectral_clustering
                    ├── tx_train.json
                    └── ...
        </pre>


### Checkpoints
We provide pretrained weights for each dataset.

Faster RCNN (baseline)
| Datasets | $\rm{AP}_{known}$ | $\rm{AP}_{unk}$ | models | config |
|:---:|:---:|:---:|:---:|:---:|
| OpenImages-Animal  | 37.8 $\pm$ 3.1 | 35.3 $\pm$ 3.9 | [model](https://www.dropbox.com/scl/fo/3zd2dls8pde89whj7wfe9/AN_qrG1oFeDu7Uj8N1qJEPo?rlkey=ct10y4cpom32u8vkjhcyzx129&st=0r1bvka6&dl=0)  |
| OpenImages-Vehicle | 39.9 $\pm$ 8.7 | 17.0 $\pm$ 5.2 | [model](https://www.dropbox.com/scl/fo/cw5zhwqo9li2lpfjmflw0/AGqckNLGJhOKbn294cji1dI?rlkey=hae1c71qtm4zjqrieusbx3m1t&st=owlz84mk&dl=0) | [cfg](configs/OpenImages/vehicle) |
| CUB200             | 62.2 $\pm$ 1.0 | 24.2 $\pm$ 1.9 | [model](https://www.dropbox.com/scl/fo/wpn5h716aiwd2298aot7y/AImUfh6Tb9q34G64aoAvNvs?rlkey=h6zz8c7uv69q6xntsk5x53fi2&st=rd3213rz&dl=0) | [cfg](configs/CUB200/random)      |
| MTSD               | 50.0           |  3.1 $\pm$ 1.2 | [model](https://www.dropbox.com/scl/fo/9ucenxkyei9lz158ndi74/AC9pwYn9p1VTO63UWR3IGX4?rlkey=aobteijvyux9dzadzjgxo5m37&st=gfq52knq&dl=0) | [cfg](configs/MTSD/spclust)       |

OpenDet
| Datasets | $\rm{AP}_{known}$ | $\rm{AP}_{unk}$ | models | cfg |
|:---:|:---:|:---:|:---:|:---:|
| OpenImages-Animal  | 36.9 $\pm$ 8.1 | 33.0 $\pm$ 4.5 | [model](https://www.dropbox.com/scl/fo/xgfz6js1uk3k37l6ytmku/AMbyXTeOD3AGnI069lkLjQ4?rlkey=b0e7qwr65yfgvgkjfrqtt4zdf&st=pav98n77&dl=0) | [cfg](configs/OpenImages/animal)  |
| OpenImages-Vehicle | 38.7 $\pm$ 7.8 | 14.4 $\pm$ 3.3 | [model](https://www.dropbox.com/scl/fo/92pwbzfp9a963jom3jhb4/ADhfWdtKhCfZJtI3jK4W4ks?rlkey=wu2ynx1bk9yw55o39zb30lfyq&st=laum8tir&dl=0) | [cfg](configs/OpenImages/vehicle) |
| CUB200             | 63.3 $\pm$ 1.1 | 27.0 $\pm$ 3.0 | [model](https://www.dropbox.com/scl/fo/q5wi2y7octvn6rvs7q6m8/AOanTp2SSk4eehuUUWc0mUM?rlkey=fh8aryz8ot810ooz8fv1h4e9d&st=0wnzivnh&dl=0) | [cfg](configs/CUB200/random)      |
| MTSD               | 51.8           |  9.9 $\pm$ 3.9 | [model](https://www.dropbox.com/scl/fo/8kv28qbzg12kay7gxjg1h/APAJ4QxwFb3eWIwxYvY99Ck?rlkey=pxfk2xzlc3pmru778gx08rn38&st=dm4frt38&dl=0) | [cfg](configs/MTSD/spclust)       |


### Evaluation
To evaluate our models, run the following command;
<pre>
python tools/train_net.py --num_gpus 8 --config-file path/to/configfile --eval-only MODEL.WEIGHTS path/to/model
</pre>
