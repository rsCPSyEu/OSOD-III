import os
from itertools import product
from copy import copy

from .voc_coco import register_voc_coco
from detectron2.data import MetadataCatalog


def register_all_voc_coco(root):
    SPLITS = [
        # VOC_COCO_openset
        ("voc_coco_20_40_test", "voc_coco", "voc_coco_20_40_test"),
        ("voc_coco_20_60_test", "voc_coco", "voc_coco_20_60_test"),
        ("voc_coco_20_80_test", "voc_coco", "voc_coco_20_80_test"),

        ("voc_coco_2500_test", "voc_coco", "voc_coco_2500_test"),
        ("voc_coco_5000_test", "voc_coco", "voc_coco_5000_test"),
        ("voc_coco_10000_test", "voc_coco", "voc_coco_10000_test"),
        ("voc_coco_20000_test", "voc_coco", "voc_coco_20000_test"),

        ("voc_coco_val", "voc_coco", "voc_coco_val"),

    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_voc_coco(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


# === add Open Images splits =========================================================
_PREDEFINED_SPLITS_OpenImages = {}
_PREDEFINED_SPLITS_OpenImages['OpenImages'] = {}
for seed, sup_cat in product([100], ['animal', 'vehicle']):
    for task, order in zip(['t1', 't2', 't3', 't4'], [[1,2,3,4], [2,3,4,1], [3,4,1,2], [4,1,2,3]]):
        _PREDEFINED_SPLITS_OpenImages['OpenImages'].update(
            {
                "seed{}_{}_{}_openimages_v6_train".format(seed, sup_cat, task): 
                    ('OpenImages/train', 'OpenImages/annotations/{}/{}_train.json'.format(sup_cat, task), sup_cat, order),
                "seed{}_{}_{}_openimages_v6_validation".format(seed, sup_cat, task):
                    ('OpenImages/validation', 'OpenImages/annotations/{}/{}_validation.json'.format(sup_cat, task), sup_cat, order),
                "seed{}_{}_{}_openimages_v6_test".format(seed, sup_cat, task):
                    ('OpenImages/test', 'OpenImages/annotations/{}/all_task_test.json'.format(sup_cat), sup_cat, order),
            }
        )

from .register_openimages import register_openimages_instances
def register_all_openimages(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_OpenImages.items():
        for key, (image_root, json_file, sup_cat, order) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_openimages_instances(
                key,
                # _get_builtin_metadata(dataset_name),
                {},
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
                sup_cat,
                order,
            )


# === add CUB200 splits =========================================================
from .register_CUB200 import register_CUB200_instances
_PREDEFINED_SPLITS_CUB200 = {}
_PREDEFINED_SPLITS_CUB200['COCO_CUB200'] = {}

split_range = list(range(1, 5)) # 1 - 4

def add_t(list_):
    return ['t{}'.format(i) for i in list_]

def exclude(t, all_):
    all_ = copy(all_)
    all_.remove(t)
    return all_
    
# train
_PREDEFINED_SPLITS_CUB200['COCO_CUB200'].update( 
    {
        'CUB200_random_t{}_train'.format(i): \
        (
            'CUB_200_2011/images',
            'CUB_200_2011/random_separation/{}_train.json'.format('_'.join(add_t( exclude(i, split_range) ))),
            exclude(i, split_range)+[i]
        ) for i in split_range
    }  
)
# val
_PREDEFINED_SPLITS_CUB200['COCO_CUB200'].update( 
    {
        'CUB200_random_t{}_val'.format(i): \
        (
            'CUB_200_2011/images', 
            'CUB_200_2011/random_separation/{}_validation.json'.format('_'.join(add_t( exclude(i, split_range) ))),
            exclude(i, split_range)+[i]
        ) for i in split_range
    }  
)
# test
_PREDEFINED_SPLITS_CUB200['COCO_CUB200'].update( 
    {
        'CUB200_random_t{}_test'.format(i): \
        (
            'CUB_200_2011/images', 
            'CUB_200_2011/random_separation/all_task_test.json',
            exclude(i, split_range)+[i]
        ) for i in split_range
    }  
)

def register_all_CUB(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_CUB200.items():
        for key, (image_root, json_file, task_order) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_CUB200_instances(
                key,
                {},
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
                task_order,
            )


# === add MTSD splits =========================================================
from .register_MTSD import register_MTSD_instances
_PREDEFINED_SPLITS_MTSD = {}
_PREDEFINED_SPLITS_MTSD['COCO_MTSD'] = {}

name_jsonname_order = [
    # name : train_test
    # jsonname: name of json file for each stage
    # order: order to load categories
    ('t1_t2', {'train': 't1', 'val': 't1', 'test': 'all_task_test'}, [1,2,3]), # (name, jsonname, order) 
    ('t1_t3', {'train': 't1', 'val': 't1', 'test': 'all_task_test'}, [1,3,2]),
    ('t1_t2t3', {'train': 't1', 'val': 't1', 'test': 'all_task_test'}, [1,2,3]),
    # ('t1t2_t3', {'train': 't1_t2', 'val': 't1_t2', 'test': 'all_task_test'}, [1,2,3]),
    # ('t1t3_t2', {'train': 't1_t3', 'val': 't1_t3', 'test': 'all_task_test'}, [1,3,2]),
]
# train
_PREDEFINED_SPLITS_MTSD['COCO_MTSD'].update( 
    {
        'MTSD_SPCLUST_{}_train'.format(name): \
        (
            'Mapillary_Traffic_Sign/images', 
            'Mapillary_Traffic_Sign/spectral_clustering/{}_train.json'.format(jsonname['train']),
            order
        ) for name, jsonname, order in name_jsonname_order
    }  
)
# val
_PREDEFINED_SPLITS_MTSD['COCO_MTSD'].update( 
    {
        'MTSD_SPCLUST_{}_val'.format(name): \
        (
            'Mapillary_Traffic_Sign/images', 
            'Mapillary_Traffic_Sign/spectral_clustering/{}_validation.json'.format(jsonname['val']),
            order
        ) for name, jsonname, order in name_jsonname_order
    }  
)
# test
_PREDEFINED_SPLITS_MTSD['COCO_MTSD'].update( 
    {
        'MTSD_SPCLUST_{}_test'.format(name): \
        (
            'Mapillary_Traffic_Sign/images', 
            'Mapillary_Traffic_Sign/spectral_clustering/{}.json'.format(jsonname['test']),
            order
        ) for name, jsonname, order in name_jsonname_order
    }  
)
# train with add_othersign
_PREDEFINED_SPLITS_MTSD['COCO_MTSD'].update( 
    {
        'MTSD_SPCLUST_{}_train_add_othersign'.format(name): \
        (
            'Mapillary_Traffic_Sign/images', 
            'Mapillary_Traffic_Sign/spectral_clustering/{}_train_add_othersign.json'.format(jsonname['train']),
            order
        ) for name, jsonname, order in name_jsonname_order
    }  
)

# test with add_othersign
_PREDEFINED_SPLITS_MTSD['COCO_MTSD'].update( 
    {
        'MTSD_SPCLUST_{}_test_add_othersign'.format(name): \
        (
            'Mapillary_Traffic_Sign/images', 
            'Mapillary_Traffic_Sign/spectral_clustering/{}_add_othersign.json'.format(jsonname['test']),
            order
        ) for name, jsonname, order in name_jsonname_order
    }  
)

def register_all_MTSD(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_MTSD.items():
        for key, (image_root, json_file, task_order) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_MTSD_instances(
                key,
                {},
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
                task_order,
            )

if __name__.endswith(".builtin"):
    # Register them all under "./datasets"
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_voc_coco(_root)
    register_all_openimages(_root)
    register_all_MTSD(_root)
    register_all_CUB(_root)
