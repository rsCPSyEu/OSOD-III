# OSOD-III 


## Overview
This is an official repository for **"Rectifying Open-Set Object Detection: Proper Evaluation and a Taxonomy"** (a submission to the 37th Conference on Neural Information Processing Systems Datasets and Benchmarks Track).

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
For downloading our annotation files, please access [this link](https://www.dropbox.com/sh/ciw4dhy4dpcqptb/AACxgUcoT4cYfUCIQKfRB-INa?dl=0).


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

- Setup the environment
```
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

# [optional] if the build process does not work well, change the version of setuptools may help you.
# conda install -c conda-forge setuptools=42
```

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
We provide pretrained weights soon.

| methods | $AP_{known}$ | $AP_{unk}$ | models | config |
|:---:|:---:|:---:|:---:|:---:|
| Faster RCNN |  |  |  |  |
| OpenDet     |  |  |  |  |


### Evaluation
To evaluate our models, run the following command;
```
python tools/train_net.py --num_gpus 8 --config-file path/to/configfile --eval-only MODEL.WEIGHTS path/to/model
```