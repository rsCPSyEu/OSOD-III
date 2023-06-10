# OSOD-III 

## Overview
This is an official repository for a preprint **Rectifying Open-set Object Detection: A Taxonomy, Practical Applications, and Proper Evaluation**.
We provide links to benchmark datasets used in the experiments.

We prepare for three datasets; 
* [Open Images v6](https://storage.googleapis.com/openimages/web/download_v6.html)
* [Caltech-UCSD Birds-200-2011 (CUB200)](https://www.vision.caltech.edu/datasets/cub_200_2011/)
* [Mapillary Traffic Sign Dataset (MTSD)](https://www.mapillary.com/dataset/trafficsign)


## Downloads 
Our datasets constructs with the original images (without any modifications) and redesigned annotation files.

- [Images]
    - Please access to the official URL (listed above) and download them in your environment.
- [Annotations]
    - Please access to our dropbox [here](https://www.dropbox.com/sh/ciw4dhy4dpcqptb/AACxgUcoT4cYfUCIQKfRB-INa?dl=0) to get the annotation files.


## Usage
All annotation data follow [MSCOCO](https://cocodataset.org/#home) format.
We can use pycocotools to load these annotation files like:
```
from pycocotools.coco import COCO
cub200 = COCO('path/to/annotaion/tX_train.json') # this instance can be used as coco_api
```


Each datasets contain some **splits** and corresponding annotations.
In each split, known and unknown classes are different. 
To check the list of known categories, please see the file ```category_X.txt```.


> Open Images
- Follow instructions in [Open Images v6](https://storage.googleapis.com/openimages/web/download_v6.html) and download original data.
- Place the dataset as followings:
```
- datasets
    - OpenImages
        - train
            - xxx.png
            - ...
        - validation
        - test

        - annotations
            - animal
                - tx_train.json
                - ...
            - vehicle
                - ...
```


> CUB200
- Follow instructions in [Caltech-UCSD Birds-200-2011 (CUB200)](https://www.vision.caltech.edu/datasets/cub_200_2011/) and download original data.
- Place the dataset as followings:
```
- datasets
    - CUB_200_2011
        - images
            - 001.Black_footed_Albatross
            - 002. ...
        - random_separation
            - tx_train.json
            - ...
```


> MTSD
- Follow instructions in [Mapillary Traffic Sign Dataset (MTSD)](https://www.mapillary.com/dataset/trafficsign) and download original data.
- Place the dataset as followings:
```
- datasets
    - Mapillary_Traffic_Sign
        - images
            - xxx.jpg
            - ...
        - spectral_clustering
            - tx_train.json
            - ...
```
