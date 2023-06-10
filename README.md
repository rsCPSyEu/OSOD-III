# OSOD-III 

## Overview
This is an official repository for **Rectifying Open-Set Object Detection: Proper Evaluation and a Taxonomy** (a submission to the 37th Conference on Neural Information Processing Systems Datasets and Benchmarks Track).

We provide access to three datasets employed in our experiments;
[Open Images v6](https://storage.googleapis.com/openimages/web/download_v6.html), [Caltech-UCSD Birds-200-2011 (CUB200)](https://www.vision.caltech.edu/datasets/cub_200_2011/), and [Mapillary Traffic Sign Dataset (MTSD)](https://www.mapillary.com/dataset/trafficsign).

## Downloads
Our datasets have been redesigned specifically for the new OSOD-III task, utilizing existing datasets.

To download the original images, please access to the original resources and follow the instructions;
- [Open Images v6](https://storage.googleapis.com/openimages/web/download_v6.html)
- [Caltech-UCSD Birds-200-2011 (CUB200)](https://www.vision.caltech.edu/datasets/cub_200_2011/)
- [Mapillary Traffic Sign Dataset (MTSD)](https://www.mapillary.com/dataset/trafficsign)

For downloading and extracting the annotation files, please access [this link](https://www.dropbox.com/sh/ciw4dhy4dpcqptb/AACxgUcoT4cYfUCIQKfRB-INa?dl=0) and follow the instruction below.

### Open Images
- Download the original images from [Open Images v6](https://storage.googleapis.com/openimages/web/download_v6.html).
- Place the dataset as follows:
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

### CUB200
- Download the original images from [Caltech-UCSD Birds-200-2011 (CUB200)](https://www.vision.caltech.edu/datasets/cub_200_2011/).
- Place the dataset as follows:
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

### MTSD
- Download the original images from [Mapillary Traffic Sign Dataset (MTSD)](https://www.mapillary.com/dataset/trafficsign).
- Place the dataset as follows:
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

## Usage
Each datasets are separeted into some **splits** according to the known/unknown classes.
To check the list of known categories, please see the file ```category_X.txt```.

All annotation data follow [MSCOCO](https://cocodataset.org/#home) format.
We can use pycocotools to load these annotation files like:
```
from pycocotools.coco import COCO
cub200 = COCO('path/to/annotaion/tX_train.json') # this instance can be used as coco_api
```