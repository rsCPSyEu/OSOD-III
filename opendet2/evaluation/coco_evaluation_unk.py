# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt
from collections import OrderedDict
import tempfile
import pycocotools.mask as mask_util
import torch
from torch.distributions.weibull import Weibull
from torch.distributions.transforms import AffineTransform
from torch.distributions.transformed_distribution import TransformedDistribution
from fvcore.common.file_io import PathManager

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from tabulate import tabulate

from functools import lru_cache

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation.fast_eval_api import COCOeval_opt
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import create_small_table

from .evaluator import DatasetEvaluator
# from detectron2.evaluation import DatasetEvaluator

from collections import defaultdict
from copy import deepcopy

# from .entropy import get_entropy_hist
# from .sigmoid_eval import evaluate_sigmoid

# from .unknown_postprocess import UnknownMetric

def get_api_result_handler(log_file):
    fhandler = logging.FileHandler(log_file, 'w')
    fhandler.setLevel(logging.INFO)
    return fhandler

class COCOEvaluator_unk(DatasetEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None, *, use_fast_impl=True):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                Although the results should be very close to the official implementation in COCO
                API, it is still recommended to compute results with the official API for use in
                papers.
        """
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir
        os.makedirs(self._output_dir, exist_ok=True)
        self._use_fast_impl = use_fast_impl
        
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        rhandler = get_api_result_handler(os.path.join(self._output_dir, 'test.log'))
        self._logger.addHandler(rhandler)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.info(
                f"'{dataset_name}' is not registered by `register_coco_instances`."
                " Therefore trying to convert it to COCO format ..."
            )
            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_json(dataset_name, cache_path)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        self._kpt_oks_sigmas = cfg.TEST.KEYPOINT_OKS_SIGMAS
        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset

        self.id2cat = {v['id']:v['name'] for k,v in self._coco_api.cats.items()} # original cats order
        self.cat2id = {v: k for k,v in self.id2cat.items()}

        # add owod detection's evaluation setting from "pascal_voc_evalution.py"
        self._dataset_name = dataset_name
        self._class_names = self._metadata.thing_classes # ordered target + non-target category names
        self.id_map = self._metadata.thing_dataset_id_to_contiguous_id # convert category-id into index in self._class_names
        
        # if ratio_ud, add extra category to be unknown
        
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            self.reverse_id_map = { # from index to category_id
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }

        self._is_2007 = False
        if cfg is not None:
            self.prev_intro_cls = cfg.OWOD.PREV_INTRODUCED_CLS # 0
            self.curr_intro_cls = cfg.OWOD.CUR_INTRODUCED_CLS # 230
            
            # cfg.OWOD.EVAL_TARGET_CLS: Max num categories to learn in the experiment (include future task).
            # e.g., if you learn 20 categories per a task and continue 4 tasks (20, 20, 20, 20), cfg.OWOD.EVAL_TARGET_CLS sould be 80.
            # This value is possibly different from cfg.MODEL.ROI_HEADS.NUM_CLASSES. 
            if cfg.OWOD.EVAL_TARGET_CLS > -1:
                self.eval_index_range = range(0, cfg.OWOD.EVAL_TARGET_CLS)
            else:
                self.eval_index_range = None # default

            self.total_num_class = cfg.MODEL.ROI_HEADS.NUM_CLASSES
            self.total_num_known_class = cfg.MODEL.ROI_HEADS.NUM_KNOWN_CLASSES           
            # ===
            # When use ratio_ud, the number of NUM_CLASSES are decreased by 1. thus, adjust self.unknown_class_index
            # self.unknown_class_index = self.total_num_class - 1 # 402 - 1 = 401
            self.unknown_class_index = self.id_map[self.cat2id['unknown']] # 401
            # ===
            self.unknown_class_id = self.cat2id['unknown']
            self.num_seen_classes = self.prev_intro_cls + self.curr_intro_cls
            self.known_classes = self._class_names[:self.num_seen_classes] # ordered
            self.known_classes_ids = [self.cat2id[c] for c in self.known_classes] # ordered
            
            if self.total_num_class == self.total_num_known_class:
                self.reverse_id_map[self.total_num_class] = self.unknown_class_id
            
            # param_save_location = os.path.join(cfg.OUTPUT_DIR, 'energy_dist_'+str(self.num_seen_classes) + '.pkl')
            # self.energy_distribution_loaded = False
            # if os.path.isfile(param_save_location) and os.access(param_save_location, os.R_OK):
            #     self._logger.info('Loading energy distribution from ' + param_save_location)
            #     params = torch.load(param_save_location)
            #     unknown = params[0]
            #     known = params[1]
            #     self.unk_dist = self.create_distribution(unknown['scale_unk'], unknown['shape_unk'], unknown['shift_unk'])
            #     self.known_dist = self.create_distribution(known['scale_known'], known['shape_known'], known['shift_known'])
            #     self.energy_distribution_loaded = True
            # else:
            #     self._logger.info('Energy distribution is not found at ' + param_save_location)

        self.paper_measure_save_path = os.path.join(self._output_dir, 'paper_measure.txt')        
        if hasattr(self._metadata, 'task_order'):
            self.task_order = self._metadata.task_order
        else:
            self.task_order = []
        
        # if true, remove overlapping predictions between known-unknown on evaluation    
        self.rmovl = cfg.OWOD.RM_OVERLAP
        self.rmovl_conf_th = cfg.OWOD.RMOVL_CONF_TH # compare only the predictions whose confidence are above this th
        self.rmovl_iou_th = cfg.OWOD.RMOVL_IOU_TH # remove if iou btw known and unknown predictions are higher than this th


    def reset(self):
        self._predictions = []
        self._predictions_voc = defaultdict(list)  # class name -> list of prediction strings
        self._predictions_for_osr = defaultdict(list) 
        self._img_order = []
        self._rpn_results = []

    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = ("bbox",)
        if cfg.MODEL.MASK_ON:
            tasks = tasks + ("segm",)
        if cfg.MODEL.KEYPOINT_ON:
            tasks = tasks + ("keypoints",)
        return tasks

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        # input : dict_keys(['file_name', 'height', 'width', 'image_id', 'image'])
        assert inputs.__len__() == 1
        for input, output in zip(inputs, outputs):
            # print('image id: {}'.format(input['image_id']))
            prediction = {"image_id": input["image_id"]}

            self._img_order.append(input["image_id"])
            
            per_image = [[] for _ in range(self._class_names.__len__())]
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                # update label based on energy
                classes = instances.pred_classes.tolist() # index
                # logits = instances.logits
                # classes_aft_eng = self.update_label_based_on_energy(logits, classes)
                # instances.pred_classes = torch.LongTensor(classes_aft_eng)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])

                # process as pascal_voc format
                image_id = input["image_id"]
                boxes = instances.pred_boxes.tensor.numpy()
                scores = instances.scores.tolist()
                classes = instances.pred_classes.tolist() # index
                
                for box, score, cls in zip(boxes, scores, classes):
                    if cls == -100:
                        continue
                    xmin, ymin, xmax, ymax = box
                    # The inverse of data loading logic in `datasets/pascal_voc.py`
                    xmin += 1
                    ymin += 1
                    self._predictions_voc[cls].append(
                        f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                    )

                    cat_id = self.reverse_id_map[cls] # from index to cat_id
                    per_image[cat_id-1].append( # -1 to convert into index
                        np.concatenate((box, [score])).astype(np.float32)
                    )  
                # per_image = [ [np.concatnate, np.concatnate, np.concatnate,... ], [], [], ...  ]
                per_image = [np.stack(per_cat).astype(np.float32) if per_cat.__len__() > 0 else [] for per_cat in per_image ]

            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)

            self._predictions.append(prediction)
            self._predictions_for_osr[input['image_id']].extend(
                per_image
            )


    # === from pascal_voc_evaluation
    def compute_avg_precision_at_many_recall_level_for_unk(self, precisions, recalls, passed_cls_ids):
        precs = {}
        for r in range(1, 10):
            r = r/10
            p = self.compute_avg_precision_at_a_recall_level_for_unk(precisions, recalls, passed_cls_ids, recall_level=r)
            precs[r] = p
        return precs

    def compute_avg_precision_at_a_recall_level_for_unk(self, precisions, recalls, passed_cls_ids, recall_level=0.5):
        precs = {}
        for iou, recall in recalls.items():
            prec = []
            # for cls_id, rec in enumerate(recall):
            assert recall.__len__() == passed_cls_ids.__len__()
            for r_idx, (rec, cls_id) in enumerate(zip(recall, passed_cls_ids)):
                if cls_id == self.unknown_class_id and len(rec)>0: # consider only "unknown" category
                    p = precisions[iou][r_idx][min(range(len(rec)), key=lambda i: abs(rec[i] - recall_level))]
                    prec.append(p)
            if len(prec) > 0:
                precs[iou] = np.mean(prec)
            else:
                precs[iou] = 0
        return precs

    def compute_WI_at_many_recall_level(self, recalls, tp_plus_fp_cs, fp_os, passed_cls_ids):
        wi_at_recall = {}
        # for r in range(1, 10):
        #     r = r/10 # [0.1, 0.2, 0.3, ..., 0.9]
        for r in np.arange(0.05, 1, 0.05):
            r = round(r, 2) # 0.XX
            wi = self.compute_WI_at_a_recall_level(recalls, tp_plus_fp_cs, fp_os, passed_cls_ids, recall_level=r)
            wi_at_recall[r] = wi
        return wi_at_recall

    def compute_WI_at_a_recall_level(self, recalls, tp_plus_fp_cs, fp_os, passed_cls_ids, recall_level=0.5):
        wi_at_iou = {}
        for iou, recall in recalls.items():
            tp_plus_fps = []
            fps = []
            # for cls_idx, rec in enumerate(recall):
            assert recall.__len__() == passed_cls_ids.__len__()
            for r_idx, (rec, cls_id) in enumerate(zip(recall, passed_cls_ids)):
                if cls_id in self.known_classes_ids and len(rec) > 0: # do not consider "unknown" category in WI
                    index = min(range(len(rec)), key=lambda i: abs(rec[i] - recall_level)) # take a index that has nearest recall_level
                    tp_plus_fp = tp_plus_fp_cs[iou][r_idx][index]
                    tp_plus_fps.append(tp_plus_fp)
                    fp = fp_os[iou][r_idx][index]
                    fps.append(fp)
            if len(tp_plus_fps) > 0:
                wi_at_iou[iou] = np.mean(fps) / np.mean(tp_plus_fps) # not (np.array(fps) / np.array(tp_plus_fps)).mean() ？
            else:
                wi_at_iou[iou] = 0
        return wi_at_iou
        
    # === 
    
    # def _unk_det_as_postprocess(self, cfg, coco_results):
        
    #     if cfg.MSP.TH > 0.:
    #         criterion = 'msp'
    #         threshold = cfg.MSP.TH
    #     elif cfg.RATIO_UD.UNK_RATIO > 0.:
    #         criterion = 'ratio'
    #         threshold = cfg.RATIO_UD.UNK_RATIO
    #     elif cfg.ENTROPY_TH > 0.:
    #         criterion = 'entropy'
    #         threshold = cfg.ENTROPY_TH
    #     else:
    #         return coco_results
        
    #     unk_checker = UnknownMetric(criterion, threshold, unk_ind=self.unknown_class_id)
    #     for a in coco_results:
    #         score, score_dist, label = a['score'], a['score_dist'], a['category_id']
    #         score = torch.tensor([score])[None, :]
    #         score_dist = torch.tensor(score_dist)[None, :]
    #         label = torch.tensor([label])[None, :]
    #         score, score_dist, label = unk_checker(score, score_dist, label)
    #         a['score'] = score[0].item()
    #         a['score_dist'] = score_dist[0].tolist()
    #         a['category_id'] = label[0].item()   
    #         a['fg_bg_score'] = score_dist[0].tolist() # if the classifier (i.e. TransFG) does not have bg class, this is incorrect
    #     return coco_results
    
    def get_category_splits(self, known_cats, start, end):
        """a function to categorise all unknown (i.e. "not known") classes into three types: valid, invalid, and others 
        Args:
            known_cats (list(str)): class names of known classes
            start (int): index of the first unknown class 
            end (int): index of the last unknown class

        Returns:
            valid_unknown_cats (list(str)): class names of unknown classes which are evaluated as unknown in this process
            invalid_unknown_cats (list(str)): class names of unknown classes which are 
            others_unknown_cats (list(str)): class names of unknown classes which are 
        """        
        target_cats = self._class_names[
            min(self.eval_index_range): max(self.eval_index_range)+1
        ]
        all_cats = self._class_names
        others_unknown_cats = list(set(all_cats) - set(target_cats))
        if start is not None and end is not None:
            valid_unknown_cats = target_cats[start:end]
        else:
            valid_unknown_cats = list(set(target_cats) - set(known_cats)) # split_all
        invalid_unknown_cats = list(
            set(all_cats) - set(known_cats) - set(valid_unknown_cats) - set(others_unknown_cats)
        )
        
        if valid_unknown_cats.__len__() == 0:
            valid_unknown_cats = None
        if invalid_unknown_cats.__len__() == 0:
            invalid_unknown_cats = None
        if others_unknown_cats.__len__() == 0:
            others_unknown_cats = None
        
        print('valid_unknown: {}'.format(valid_unknown_cats))
        print('invalid_unknown: {}'.format(invalid_unknown_cats))
        # print('others_unknown_cats: {}'.format(others_unknown_cats))
        
        return valid_unknown_cats, invalid_unknown_cats, others_unknown_cats
    
    
    def convert_to_unknown(self, coco_gt, known_cat_ids, 
                        valid_unknown_cat_ids, invalid_unknown_cat_ids,
                        others_unknown_cat_ids, unknown_cat_id,  
                        others_attr={'keep': False, 'ignore':False}):
        """
        In pycocotools, 
        convert unknown GTs (for current task) into "unknown", 
        and add "iscrowd" tag to ignore non-interested unknown GTs in evaluation. 
        """
        print('=== Converting some GTs to unknown / setting iscrowd-tag to be ignored ===')
        
        # label unseen (for current task) category GTs into "unknown"
        unk_info = coco_gt.cats[unknown_cat_id]
        if 'freebase_id' in unk_info:
            unk_fbid = unk_info['freebase_id']

        add_unk = 0
        add_iscrowd = 0
        deleted = 0
        num_ann = coco_gt.anns.__len__()
        org_gt_anns = copy.deepcopy(coco_gt.anns) # keep original category id's annotation
        
        new_coco_gt_anns = dict()
        for k in coco_gt.anns.keys():
            cat_id = coco_gt.anns[k]['category_id']
            # if out of interested categories in whole experiment, we want to exclude those annotation.
            # otherwise, performance (i.e., mAP) of "unknown" will be affected by those (non-interested) unknown GTs. 
            # To exclude them, utilize "iscrowd" tag used in pycocotools. "iscrowd" GTs are ignored in evaluation (counted as neither TP nor FP).
            if cat_id not in known_cat_ids:
                # others (keep or not / ignore or not)
                if cat_id in others_unknown_cat_ids:
                    if not others_attr['keep']:
                        deleted += 1
                        continue
                    else:
                        if others_attr['ignore']:
                            coco_gt.anns[k]['iscrowd'] = True
                            add_iscrowd += 1
                        else:
                            coco_gt.anns[k]['iscrowd'] = False
                
                if cat_id in invalid_unknown_cat_ids:
                    coco_gt.anns[k]['iscrowd'] = True
                    add_iscrowd += 1
                    
                coco_gt.anns[k]['category_id'] = unknown_cat_id
                add_unk += 1
                if 'freebase_id' in unk_info and 'freebase_id' in coco_gt.anns[k]:
                    coco_gt.anns[k]['freebase_id'] = unk_fbid
                
                new_coco_gt_anns[k] = coco_gt.anns[k] # add to new anns
            else:
                assert cat_id in known_cat_ids
                new_coco_gt_anns[k] = coco_gt.anns[k]
        
        print('=== added {} / {} GTs as unknown ==='.format(add_unk, coco_gt.anns.__len__()))
        print('=== added {} / {} GTs as iscrowd=True ==='.format(add_iscrowd, coco_gt.anns.__len__()))
        print('=== deleted {} / {} GTs as others ==='.format(deleted, num_ann))
        print()
        # coco_gt.anns = new_coco_gt_anns
        coco_gt.dataset['annotations'] = list(new_coco_gt_anns.values())
        coco_gt.createIndex()
        return coco_gt, org_gt_anns
    

    def get_PRcurve(self, x, y, area_names=['all']):
        '''
        x : [101]
        y : [101, cats, 4]
        '''
        colors = {
            'all': 'red',
            'small': 'orange',
            'medium': 'blue',
            'large': 'green'
        }
        for a_idx, a in enumerate(area_names):
            y_area = y[:, :, a_idx]
            y_area = np.array([np.mean(ya[ya>-1]) for ya in y_area]) # mean of category
            plt.plot(x, y_area, label=a, c=colors[a])
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.grid()
        plt.legend()
        return


    def plot_PRcurve(self, eval, maxDets=100, cname='all', savepath=None):    
        precisions = eval['precision'] # [10, 101, cats, 4, 3]
        recalls = eval['recall'] # [10, cats, 4, 3]

        maxDets_names = eval['params'].maxDets
        maxDets_idx = maxDets_names.index(maxDets)
        area_names = eval['params'].areaRngLbl
        iou_thresh = eval['params'].iouThrs
        rec_thresh = eval['params'].recThrs
        
        for iou_idx, iou in enumerate(iou_thresh.tolist()):
            striou = str(round(iou, 2)).replace('.', '')
            y = precisions[iou_idx, :, :, :, maxDets_idx]
            x = rec_thresh
            self.get_PRcurve(x, y, area_names)
            plt.title('IOU @ {}'.format(round(iou, 2)))
            if savepath is not None:
                pltpath = os.path.join(savepath,
                                       '{}_{}.png'.format(cname, striou))
                plt.savefig(pltpath)
            else:
                plt.show()
            plt.clf()
        return


    def save_osr(self):
        # gather predictions_for_osr
        if self._distributed:
            comm.synchronize()
            all_predictions_for_osr = comm.gather(self._predictions_for_osr, dst=0)
            # print('all_predictions_for_osr: {}'.format(all_predictions_for_osr.__len__()))
            # print('{}'.format([a.__len__() for a in all_predictions_for_osr]))
            
            predictions_for_osr = defaultdict(list)
            for predictions_per_rank in all_predictions_for_osr:
                # print('predictions_per_rank: {}'.format(predictions_per_rank.__len__()))
                for imgid, lines in predictions_per_rank.items():
                    predictions_for_osr[imgid].extend(lines)
            del all_predictions_for_osr
              
            if not comm.is_main_process():
                return {}
        else:
            predictions_for_osr = self._predictions_for_osr
            
        # save mmdetection cfm result
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            img_order = list(predictions_for_osr.keys())
            predictions_for_osr = [
                predictions_for_osr[img_id] for img_id in img_order
            ]
            file_path = os.path.join(self._output_dir, "result.pkl")
            with PathManager.open(file_path, "wb") as f:
                pickle.dump(
                    {
                        'img_id': img_order,
                        'detection': predictions_for_osr,
                    }, f
                )
        return 
    
    
    def evaluate_rpn(self, file_path, iouThrs=np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])):
        """evaluate rpn output

        Args:
            rpn_results (_type_): list of detections from RPN
            eval_rpn (_type_): path to rpn result file

        Returns:
            _type_: _description_
        """        
        if self._distributed:
            comm.synchronize()
            rpn_results = comm.gather(self._rpn_results, dst=0)
            rpn_results = list(itertools.chain(*rpn_results))
            if not comm.is_main_process():
                return
        else:
            rpn_results = self._rpn_results
        
        if len(rpn_results) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return
            
        print('save {} ({})'.format(file_path, rpn_results.__len__()))
        with open(file_path, 'w') as f:
            json.dump(rpn_results, f)
                    
        # logger 
        from pathlib import Path
        text_file_name = os.path.join(Path(file_path).parent, 'rpn.txt')
        rhandler = get_api_result_handler(text_file_name)
        self._logger.addHandler(rhandler)
        
        # retrieve known/unknown category ids
        start, end = (len(self.known_classes), len(self.eval_index_range))
        known_cat_ids = self.known_classes_ids
        valid_unknown_cats, _, _ = self.get_category_splits(self.known_classes, start, end)
        valid_unknown_cat_ids = self._coco_api.getCatIds(catNms=valid_unknown_cats)
        
        # calculate recall for [known / unknown / all] categories
        category_ids = {
            'known': known_cat_ids,
            'unknown': valid_unknown_cat_ids,
            'all': known_cat_ids+valid_unknown_cat_ids,
        }
        
        for (key, target_category_ids) in category_ids.items():
            
            # copy coco api to keep api unchanged for ROI heads evaluation (i.e., default evalution process)
            coco_gt = deepcopy(self._coco_api)
            coco_gt.cats = {1: {'id': 1, 'name': 'foreground'}}
    
            # convert all gt into category 1 (foreground)
            new_anns = []
            for k in coco_gt.anns.keys():
                ann = coco_gt.anns[k]
                if ann['category_id'] in target_category_ids:
                    ann['category_id'] = 1
                    new_anns.append(ann)
                else:
                    continue
            coco_gt.dataset['annotations'] = list(new_anns)
            coco_gt.createIndex()
            
            coco_dt = coco_gt.loadRes(rpn_results)
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            
            # parameter setting
            coco_eval.params.catIds = [1]
            coco_eval.params.iouThrs = iouThrs
            
            coco_eval.evaluate()
            coco_eval.accumulate()
            self._logger.info('\n' + 'evaluation @ [{}]'.format(key))
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                coco_eval.summarize()
            self._logger.info(redirect_string.getvalue())
        
            del coco_gt
        
        _ = self._logger.handlers.pop(-1)
        return
    
            
    def evaluate(self, others_attrs=None, is_validation=False):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))
            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        # check whether the raio are small or not
        # self._eval_average_ratio(predictions, self._output_dir)

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            self._eval_predictions(set(self._tasks), predictions, others_attrs, is_validation=is_validation)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)


    def _eval_average_ratio(self, predictions, output_dir):
        """Evaluate mean score of top1/top2 ratio
        Lower values mean the model more tend to output uniform probabilities
        """
        num_inst = 0
        all_ratios = []
        for p in predictions:
            inst_per_img = p['instances']
            ni = inst_per_img.__len__()
            ratios = [i['ratio'] for i in inst_per_img]
            all_ratios.extend(ratios)
            num_inst += ni
        average_ratio = np.array(all_ratios).mean()
        hist, x = self.plot_ratio_histgram(all_ratios, output_dir, bins=100, range=(1,51))
        res = [
            'avg_ratio: {}'.format(average_ratio),
            'hist: {}'.format(hist.tolist()),
        ]
        with open(os.path.join(output_dir, 'avg_ratio.txt'), 'w') as f:
            f.writelines('\n'.join(res))
        return


    def plot_ratio_histgram(self, ratios, output_dir, bins=100, range=(1, 51)):
        ratios = np.array(ratios)
        hist, x = np.histogram(ratios, bins=bins, range=range)
        title = 'bins={} / range={}'.format(bins, range)
        diff = x[1] - x[0]
        plt.bar(x[:-1]+(diff/2), hist)
        plt.title(title)
        plt.savefig(os.path.join(output_dir, 'ratio_hist.png'))

        with open(os.path.join(output_dir, 'ratio_hist.pkl'), 'wb') as f:
            pickle.dump(
                {
                    'data': ratios,
                    'hist': hist,
                    'x': x,
                    'bins': bins,
                    'range': range
                }, 
                f
            )
        return hist, x


    # def _calc_entropy(self, coco_results, mode):
    #     save_dir = os.path.join(self._output_dir, 'entropy_hist', mode)
    #     os.makedirs(save_dir, exist_ok=True)
    #     entropies, dets, hist_ann_ids = get_entropy_hist(
    #         self._coco_api, coco_results, self.known_classes_ids, save_dir, 
    #         unk_id=self.unknown_class_id, bg_id=self.unknown_class_id+1,
    #         iou_th=0.5, max_dets=3000, hist_b=60, hist_range=(0,6), mode=mode
    #     )
    #     entropies = {k: d.data.cpu() for k,d in entropies.items()}
    #     with open(os.path.join(save_dir, 'entropies.pkl'), 'wb') as f:
    #         pickle.dump(entropies, f)
    #     with open(os.path.join(save_dir, 'samples.pkl'), 'wb') as f:
    #         pickle.dump(dets, f)
    #     with open(os.path.join(save_dir, 'sample_ids.pkl'), 'wb') as f:
    #         pickle.dump(hist_ann_ids, f)
    #     return


    def _vis_feature_embedding(self, coco_results):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from sklearn.manifold import TSNE
        import random
    
        def visualize_embeddings(x, y_pred, save_dir):
            tsne = TSNE(n_components=2, random_state=0)

            # Apply t-SNE to the feature embeddings
            x_2d = tsne.fit_transform(x)

            label_list = np.unique(y_pred)

            # Get number of unique classes
            num_classes = len(label_list)

            # Set up plot colors and markers
            # if num_classes > 20:
            #     colors = plt.cm.tab20(np.arange(num_classes))
            # else:
            colors = list(mcolors.CSS4_COLORS.values())
            
            # markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'H', '<', '>', 'd']
            marker = 'o'

            # Plot the feature embeddings
            # only known features
            plt.figure(figsize=(10, 8))
            for i in range(num_classes):
                
                # if label_list[i] == self.unknown_class_id:
                #     continue
                
                cname = self._coco_api.cats[label_list[i]]['name']
                plt.scatter(
                    x_2d[y_pred==label_list[i], 0], x_2d[y_pred==label_list[i], 1], 
                    c=colors[i], marker=marker, label=cname)
            plt.legend(loc='best')
            plt.savefig(os.path.join(save_dir, 'known.png'))
            
            # # plot unknown
            # cname = 'unknown'
            # plt.scatter(
            #     x_2d[y_pred==self.unknown_class_id, 0], x_2d[y_pred==self.unknown_class_id, 1], 
            #     c=colors[i+1], marker=marker, label=cname)
            # plt.legend(loc='best')
            # plt.savefig(os.path.join(save_dir, 'known+unk.png'))
            
        
        save_dir = os.path.join(self._output_dir, 'feat_embed')
        os.makedirs(save_dir, exist_ok=True)
        
        min_samples = 5
        sample_classes = 10
        all_cids = np.array([d['category_id'] for d in coco_results])
        hist = np.array([(all_cids==i).sum() for i in np.arange(1, self.unknown_class_id+1)])
        valid_category_index = hist[:-1] > min_samples # remove unknown index
        valid_category_ids = np.arange(1, self.unknown_class_id)[valid_category_index]
        
        valid_category_ids = random.sample(valid_category_ids.tolist(), sample_classes) + [self.unknown_class_id]
        embeds = []
        cids = []
        for d in coco_results:
            emb = torch.Tensor(d['embed'])
            cid = torch.Tensor([d['category_id']])
            if cid in valid_category_ids:
                embeds.append(emb)
                cids.append(cid)
        embeds = torch.stack(embeds, dim=0)
        cids = torch.cat(cids).long()
        visualize_embeddings(embeds, cids, save_dir)
        return

    

    # def _calc_sigmoid_accuracy(self, coco_gt, coco_results, meta):
    #     evaluate_sigmoid(coco_gt, coco_results, meta, save_dir=self._output_dir)
    #     return


    def _eval_predictions_from_file(self, coco_results, others_attrs):
        self._results = OrderedDict()
        tasks = set(self._tasks)
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            # reverse_id_mapping = { # from index to category_id
            #     v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            # }
            # if self.total_num_class == self.total_num_known_class:
            #     reverse_id_mapping[self.total_num_class] = self.unknown_class_id
            reverse_id_mapping = self.reverse_id_map
            
        if not comm.is_main_process():
            return {}
        
        # === 
        # if self.eval_entropy:
        #     self._logger.info('=== Start entropy evaliation ===')
        #     for mode in ['gt', 'dt']:
        #         self._calc_entropy(coco_results, mode=mode)
        #     self._logger.info('=== End entropy evaliation ===\n')

        # if self.eval_sigmoid:
        #     self._logger.info('=== Start sigmoid evaliation ===')
        #     # if os.path.exists(os.path.join(self._output_dir, 'coco_instances_results.json')):
        #     #     with open(os.path.join(self._output_dir, 'coco_instances_results.json'), 'r') as f:
        #     #         coco_results = json.load(f)
        #     self._calc_sigmoid_accuracy(self._coco_api, coco_results, self._metadata)
        #     self._logger.info('=== End sigmoid evaliation ===\n')

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )
        for task in sorted(tasks):
            for others_attr in others_attrs:
                coco_eval = (
                    self._evaluate_predictions_on_coco(
                        coco_results,
                        task,
                        known_cat_ids = self.known_classes_ids,
                        unknown_cat_id = reverse_id_mapping[self.unknown_class_index],
                        kpt_oks_sigmas=self._kpt_oks_sigmas,
                        use_fast_impl=self._use_fast_impl,
                        others_attr = others_attr,
                    )
                    if len(coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )
                res = self._derive_coco_results(
                    coco_eval, task, 
                    class_names=self._metadata.get("thing_classes")
                )
            self._results[task] = res
        return copy.deepcopy(self._results)


    def _eval_predictions(self, tasks, predictions, others_attrs, is_validation=False):
        """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            # reverse_id_mapping = { # from index to category_id
            #     v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            # }
            reverse_id_mapping = self.reverse_id_map
            for result in coco_results:
                category_id = result["category_id"]
                assert (
                    category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()        
    
    
        # additional evaluation process 
        # if self.eval_entropy:
        #     self._logger.info('=== Start entropy evaliation ===')
        #     for mode in ['gt', 'dt']:
        #         self._calc_entropy(coco_results, mode=mode)
        #     self._logger.info('=== End entropy evaliation ===\n')

        # if self.eval_feat_embed:
        #     self._logger.info('=== Start feature embedding visualization ===')
        #     self._vis_feature_embedding(coco_results)
        #     self._logger.info('=== End feature embedding visualization ===\n')

        # if self.eval_sigmoid:
        #     self._logger.info('=== Start sigmoid evaliation ===')
        #     # if os.path.exists(os.path.join(self._output_dir, 'coco_instances_results.json')):
        #     #     with open(os.path.join(self._output_dir, 'coco_instances_results.json'), 'r') as f:
        #     #         coco_results = json.load(f)
        #     self._calc_sigmoid_accuracy(self._coco_api, coco_results, self._metadata)
        #     self._logger.info('=== End sigmoid evaliation ===\n')


        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )
        for task in sorted(tasks):
            for others_attr in others_attrs:
                coco_eval = (
                    self._evaluate_predictions_on_coco(
                        coco_results,
                        task,
                        known_cat_ids = self.known_classes_ids,
                        unknown_cat_id = reverse_id_mapping[self.unknown_class_index],
                        kpt_oks_sigmas=self._kpt_oks_sigmas,
                        use_fast_impl=self._use_fast_impl,
                        others_attr = others_attr,
                        is_validation = is_validation,
                    )
                    if len(coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )
                res = self._derive_coco_results(
                    coco_eval, task, 
                    class_names=self._metadata.get("thing_classes")
                )
            self._results[task] = res

    def _eval_box_proposals(self, predictions):
        """
        Evaluate the box proposals in predictions.
        Fill self._results with the metrics for "box_proposals" task.
        """
        if self._output_dir:
            # Saving generated box proposals to file.
            # Predicted box_proposals are in XYXY_ABS mode.
            bbox_mode = BoxMode.XYXY_ABS.value
            ids, boxes, objectness_logits = [], [], []
            for prediction in predictions:
                ids.append(prediction["image_id"])
                boxes.append(prediction["proposals"].proposal_boxes.tensor.numpy())
                objectness_logits.append(prediction["proposals"].objectness_logits.numpy())

            proposal_data = {
                "boxes": boxes,
                "objectness_logits": objectness_logits,
                "ids": ids,
                "bbox_mode": bbox_mode,
            }
            with PathManager.open(os.path.join(self._output_dir, "box_proposals.pkl"), "wb") as f:
                pickle.dump(proposal_data, f)

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating bbox proposals ...")
        res = {}
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = _evaluate_box_proposals(predictions, self._coco_api, area=area, limit=limit)
                key = "AR{}@{:d}".format(suffix, limit)
                res[key] = float(stats["ar"].item() * 100)
        self._logger.info("Proposal metrics: \n" + create_small_table(res))
        self._results["box_proposals"] = res


    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """
        
        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        
        # precision has dims (iou, recall, cls, area range, max dets)
        # assert len(class_names) == precisions.shape[2] # order is sorted category ids ...
        assert coco_eval.params.catIds.__len__() == precisions.shape[2]

        results_per_category = []
        eval_categories_ids = coco_eval.params.catIds
        id2ind_of_prec = {id_: idx for idx, id_ in enumerate(eval_categories_ids)} # mapping from category_id to index_of_precision
        for name in class_names:
            cat_id = self.cat2id[name]
            if cat_id not in eval_categories_ids:
                continue
            idx = id2ind_of_prec[cat_id]
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)
        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results


    def _evaluate_predictions_on_coco(
        self, coco_results, iou_type, 
        known_cat_ids, unknown_cat_id, 
        kpt_oks_sigmas=None, use_fast_impl=True,
        others_attr = {'keep': False, 'ignore':False},
        is_validation = False,
    ):
        """
        Evaluate the coco results using COCOEval API.
        Args: 
            known_cat_ids : list of category ids in current task.
            unknown_cat_id: category id of "unknown" class
        """
        self._logger.info('\n=== \nothers_attr={}\n==='.format(others_attr))
        output_dir = os.path.join(self._output_dir, 'keep_{}_ign_{}'.format(others_attr['keep'], others_attr['ignore']))
        os.makedirs(output_dir, exist_ok=True)
        assert len(coco_results) > 0
        
        if iou_type == "segm":
            coco_results = copy.deepcopy(coco_results)
            # When evaluating mask AP, if the results contain bbox, cocoapi will
            # use the box area as the area of the instance, instead of the mask area.
            # This leads to a different definition of small/medium/large.
            # We remove the bbox field to let mask AP use mask area.
            for c in coco_results:
                c.pop("bbox", None)
                    
        coco_dt = self._coco_api.loadRes(coco_results)
        coco_eval = (COCOeval_opt if use_fast_impl else COCOeval)(self._coco_api, coco_dt, iou_type)
        # =================================================================
        # Remove all overlapping predictions between known and unknown and save them
        if self.rmovl:
            rmovl_save_file = os.path.join(self._output_dir, "coco_instances_results_rmovl.json")
            coco_results_rmovl = self.setup_rmovl(
                coco_eval, rmovl_save_file, 
                unk_id = unknown_cat_id,
                conf_th = self.rmovl_conf_th,
                rm_iou_th = self.rmovl_iou_th,
            )
            print('before rmovl: {}'.format(coco_results.__len__()))
            print('after rmovl: {}'.format(coco_results_rmovl.__len__()))
            coco_dt = self._coco_api.loadRes(coco_results_rmovl)
            coco_eval = (COCOeval_opt if use_fast_impl else COCOeval)(self._coco_api, coco_dt, iou_type)
        # =================================================================
                
        # evaluation per category
        text_file_name = os.path.join(output_dir, 'known.txt')
        rhandler = get_api_result_handler(text_file_name)
        self._logger.addHandler(rhandler)
        
        coco_eval.params.catIds = known_cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        self._logger.info('\n' + 'evaluation @ [all]')
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            coco_eval.summarize()
        self._logger.info(redirect_string.getvalue())
        
        pltroot = os.path.join(output_dir, 'PRC')
        os.makedirs(pltroot, exist_ok=True)
        # get PR curve 
        self.plot_PRcurve(coco_eval.eval, maxDets=100, 
                          cname='all', savepath=pltroot)
        
        if is_validation:
            return coco_eval
        
        # for cat_id in (known_cat_ids):
        #     cat_name = self._coco_api.cats[cat_id]['name'].replace(' ', '')
        #     coco_eval.params.catIds = [cat_id]
        #     coco_eval.evaluate()
        #     coco_eval.accumulate()
        #     self._logger.info('\n' + 'evaluation @ [{}]({})'.format(cat_name, cat_id))
        #     redirect_string = io.StringIO()
        #     with contextlib.redirect_stdout(redirect_string):
        #         coco_eval.summarize()
        #     self._logger.info(redirect_string.getvalue())
            
        #     # get PR curve 
        #     self.plot_PRcurve(coco_eval.eval, maxDets=100, 
        #                       cname=cat_name, savepath=pltroot)
            
        _ = self._logger.handlers.pop(-1)
        
        # === [Remove to avoid split_n evaluation] ===
        # num_splits = self.task_order.__len__()
        # num_cat_per_split = known_cat_ids.__len__()
        # if num_splits > 0:
        #     Range = [(num_cat_per_split*(i+1), num_cat_per_split*(i+2)) if i!=(num_splits-1) else (num_cat_per_split*1, num_cat_per_split*(num_splits)) \
        #             for i in range(num_splits)]
        # else:
        #     Range = [(num_cat_per_split, len(self.eval_index_range))]
        # === [Remove to avoid split_n evaluation] ===
        Range = [(len(self.known_classes), len(self.eval_index_range))]
        
        print('Range: {}'.format(Range))
        org_gt_anns = None
        for rng_idx, rng in enumerate(Range):
    
            start, end = rng
            eval_index_range = range(start, end)
            
            if rng_idx != (Range.__len__()-1):
                dir_name = 'split_{}'.format(self.task_order[rng_idx+1])
            else:
                dir_name = 'split_all' 
            split_dir = os.path.join(output_dir, dir_name)
            os.makedirs(split_dir, exist_ok=True)
            
            pltroot = os.path.join(split_dir, 'PRC')
            os.makedirs(pltroot, exist_ok=True)
            
            if org_gt_anns is not None:
                assert rng_idx > 0
                # back-convert [unknown] -> [original category]
                self._logger.info('restore original anntation at idx={}'.format(rng_idx))
                # self._coco_api.anns = org_gt_anns
                self._coco_api.dataset['annotations'] = list(org_gt_anns.values())
                self._coco_api.createIndex()
            
            redirect_string = io.StringIO()
            self._logger.info('\n@ {}'.format(dir_name))
            valid_unknown_cats, invalid_unknown_cats, others_unknown_cats = \
                self.get_category_splits(self.known_classes, start, end)
                
            # if 'other-sign' in others_unknown_cats:
            #     valid_unknown_cats.append(
            #         others_unknown_cats.pop(others_unknown_cats.index('other-sign'))
            #     )
                
            valid_unknown_cat_ids = self._coco_api.getCatIds(catNms=valid_unknown_cats)
            invalid_unknown_cat_ids = self._coco_api.getCatIds(catNms=invalid_unknown_cats)
            others_unknown_cat_ids = self._coco_api.getCatIds(catNms=others_unknown_cats)              
            
            # convert [original category] -> [unknown], if the categories are unknown in current task
            self._coco_api, org_gt_anns = self.convert_to_unknown(self._coco_api, known_cat_ids, 
                                                           valid_unknown_cat_ids = valid_unknown_cat_ids,
                                                           invalid_unknown_cat_ids = invalid_unknown_cat_ids,
                                                           others_unknown_cat_ids = others_unknown_cat_ids,
                                                           unknown_cat_id = unknown_cat_id, 
                                                           others_attr = others_attr)
            self._logger.info(redirect_string.getvalue()+'\n')

            if iou_type == "segm":
                coco_results = copy.deepcopy(coco_results)
                # When evaluating mask AP, if the results contain bbox, cocoapi will
                # use the box area as the area of the instance, instead of the mask area.
                # This leads to a different definition of small/medium/large.
                # We remove the bbox field to let mask AP use mask area.
                for c in coco_results:
                    c.pop("bbox", None)

            # evaluation per category
            text_file_name = os.path.join(split_dir, 'known+unk.txt')
            rhandler = get_api_result_handler(text_file_name)
            self._logger.addHandler(rhandler)
            
            # reload coco_eval becase self._coco_api has been changed
            coco_dt = self._coco_api.loadRes(coco_results)
            coco_eval = (COCOeval_opt if use_fast_impl else COCOeval)(self._coco_api, coco_dt, iou_type)
            
            # =================================================================
            # Remove all overlapping predictions between known and unknown and save them
            if self.rmovl:
                coco_dt = self._coco_api.loadRes(coco_results_rmovl)
                coco_eval = (COCOeval_opt if use_fast_impl else COCOeval)(self._coco_api, coco_dt, iou_type)
            # =================================================================
                        
            coco_eval.params.catIds = known_cat_ids + [unknown_cat_id] 
            coco_eval.evaluate()
            coco_eval.accumulate()
            self._logger.info('\n' + 'evaluation @ [all]')
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                coco_eval.summarize()
            self._logger.info(redirect_string.getvalue())
            
            # get PR curve 
            self.plot_PRcurve(coco_eval.eval, maxDets=100, 
                              cname='all+unk', savepath=pltroot)
            
            coco_eval.params.catIds = [unknown_cat_id]
            coco_eval.evaluate()
            coco_eval.accumulate()
            self._logger.info('\n' + 'evaluation @ [unknown]')
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                coco_eval.summarize()
            self._logger.info(redirect_string.getvalue())

            # get PR curve 
            self.plot_PRcurve(coco_eval.eval, maxDets=100, 
                              cname='unknown', savepath=pltroot)

            _ = self._logger.handlers.pop(-1)
        
        # back-convert [unknown] -> [original category]
        # self._coco_api.anns = org_gt_anns
        self._coco_api.dataset['annotations'] = list(org_gt_anns.values())
        self._coco_api.createIndex()
        
        return coco_eval
    

    def setup_rmovl(
        self, 
        coco_eval, 
        save_file,
        unk_id,
        conf_th = 0.5,
        rm_iou_th = 0.5, 
    ):
        def convert_bbox(anns, type_='np'):
            labels = [a['category_id'] for a in anns]
            bbox = [a['bbox'] for a in anns]
            if type_ == 'np':
                bbox = np.array(bbox).astype(np.float32)
                labels = np.array(labels).astype(np.int64)
            elif type_ == 'torch':
                bbox = torch.Tensor(bbox).type(torch.FloatTensor)
                labels = torch.Tensor(labels).type(torch.LongTensor)
            return bbox, labels
        
        def remove_overlap(dt_bbox, dt_labels, unk_id, rm_iou_th=0.5):
            unk_pred_ind = (dt_labels == unk_id)
            unk_bbox = dt_bbox[unk_pred_ind]
            known_bbox = dt_bbox[~unk_pred_ind]
            
            unk_index = unk_pred_ind.nonzero()[0]
            known_index = (~unk_pred_ind).nonzero()[0]
            
            if known_bbox.__len__() == 0 or unk_bbox.__len__() == 0:
                return np.concatenate((known_index, unk_index))
            
            iscrowd = [0 for _ in range(unk_bbox.__len__())]
            ious = maskUtils.iou(known_bbox, unk_bbox, iscrowd) # xywh
            
            k_max_ious = ious.max(axis=1)
            u_max_ious = ious.max(axis=0)
            
            remain_k_ind = known_index[(k_max_ious < rm_iou_th).nonzero()[0]]
            remain_u_ind = unk_index[(u_max_ious < rm_iou_th).nonzero()[0]]
            remain_ind = np.concatenate((remain_k_ind, remain_u_ind))
            return remain_ind
        
        gt = coco_eval.cocoGt
        dt = coco_eval.cocoDt
        
        rmovl_coco_results = []        
        for imgid in list(gt.imgs.keys()):            
            dt_annids = dt.getAnnIds(imgIds=[imgid])
            dt_anns = dt.loadAnns(ids=dt_annids)
            dt_anns_h = [a for a in dt_anns if a['score']>=conf_th] # high score predictions
            dt_anns_l = [a for a in dt_anns if a['score']<conf_th] # low score predictions
            
            # remove both overlapping predictions
            dt_bbox, dt_labels = convert_bbox(dt_anns_h)
            # gt_bbox, gt_labels = convert_bbox(gt_anns)
            
            remain_ind = remove_overlap(dt_bbox, dt_labels, unk_id, rm_iou_th=rm_iou_th)
            
            # remove filterd predictions 
            dt_anns_h = [a for i, a in enumerate(dt_anns_h) if i in remain_ind] # high score predictions
            new_dt_anns = dt_anns_h + dt_anns_l
            new_dt_anns = sorted(new_dt_anns, key=lambda x: x['id'])
            print('removed: {}/{}'.format(dt_anns.__len__() - new_dt_anns.__len__(), dt_anns.__len__()))
            rmovl_coco_results.extend(new_dt_anns)
        
        with open(save_file, 'w') as f:
            json.dump(rmovl_coco_results, f)
            
        return rmovl_coco_results
        
    
    def save_predictions_vos(self):
        all_predictions = comm.gather(self._predictions_voc, dst=0)
        if not comm.is_main_process():
            return
        
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions
        
        self._logger.info(
            "Evaluating {} using {} metric. "
            "Note that results do not use the official Matlab API.".format(
                self._dataset_name, 2007 if self._is_2007 else 2012
            )
        )
        if self._output_dir:
            file_path = os.path.join(self._output_dir, "prediction_voc_style.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(predictions))
                f.flush()
                
        return predictions


    def evaluate_unk(self, predictions, others_attrs):
        for others_attr in others_attrs:
            org_predictions = predictions
            _ = self.evaluate_ukn_loop(predictions, others_attr)
        return         
    
        
    def evaluate_ukn_loop(self, predictions, others_attr=None):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """ 
        self._logger.info('\n=== \nothers_attr={}\n==='.format(others_attr))
        output_dir = os.path.join(self._output_dir, 'keep_{}_ign_{}'.format(others_attr['keep'], others_attr['ignore']))
        os.makedirs(output_dir, exist_ok=True)
        
        # === [Remove to avoid split_n evaluation] ===
        # num_splits = self.task_order.__len__()
        # num_cat_per_split = self.known_classes_ids.__len__()
        # if num_splits > 0:
        #     Range = [(num_cat_per_split*(i+1), num_cat_per_split*(i+2)) if i!=(num_splits-1) else (num_cat_per_split*1, num_cat_per_split*(num_splits)) \
        #             for i in range(num_splits)]
        # else:
        #     Range = [(num_cat_per_split, len(self.eval_index_range))]
        # === [Remove to avoid split_n evaluation] ===
        Range = [(len(self.known_classes), len(self.eval_index_range))] # TODO switch when evaluate with single split
        
        org_gt_anns = None
        for rng_idx, rng in enumerate(Range): # 
            start, end = rng
            eval_index_range = range(start, end)
            
            if rng_idx != (Range.__len__()-1):
                dir_name = 'split_{}'.format(self.task_order[rng_idx+1])
            else:
                dir_name = 'split_all' 
            split_dir = os.path.join(output_dir, dir_name)
            
            os.makedirs(split_dir, exist_ok=True)
            paper_measure_save_path = os.path.join(split_dir, 'paper_measure.txt')
            
            if org_gt_anns is not None:
                assert rng_idx > 0
                # back-convert [unknown] -> [original category]
                print('restore original anntation at rng_idx={}'.format(rng_idx))
                # self._coco_api.anns = org_gt_anns
                self._coco_api.dataset['annotations'] = list(org_gt_anns.values())
                self._coco_api.createIndex()
            
            redirect_string = io.StringIO()
            self._logger.info('\n@ {}'.format(dir_name))
            valid_unknown_cats, invalid_unknown_cats, others_unknown_cats = \
                self.get_category_splits(self.known_classes, start, end)
              
            # # others_unknown_cats could be treated in various way : depending on keep/ignore flag  
            # if 'other-sign' in others_unknown_cats:
            #     valid_unknown_cats.append(
            #         others_unknown_cats.pop(others_unknown_cats.index('other-sign'))
            #     )
                
            valid_unknown_cat_ids = self._coco_api.getCatIds(catNms=valid_unknown_cats)
            invalid_unknown_cat_ids = self._coco_api.getCatIds(catNms=invalid_unknown_cats)
            others_unknown_cat_ids = self._coco_api.getCatIds(catNms=others_unknown_cats)
            # convert [original category] -> [unknown], if the categories are unknown in current task
            self._coco_api, org_gt_anns = self.convert_to_unknown(self._coco_api, 
                                                           self.known_classes_ids, 
                                                           valid_unknown_cat_ids = valid_unknown_cat_ids,
                                                           invalid_unknown_cat_ids = invalid_unknown_cat_ids,
                                                           others_unknown_cat_ids = others_unknown_cat_ids,
                                                           unknown_cat_id = self.unknown_class_id, 
                                                           others_attr = others_attr)

            with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
                res_file_template = os.path.join(dirname, "{}.txt")

                aps = defaultdict(list)  # iou -> ap per class
                recs = defaultdict(list)
                precs = defaultdict(list)
                all_recs = defaultdict(list)
                all_precs = defaultdict(list)
                unk_det_as_knowns = defaultdict(list)
                num_unks = defaultdict(list)
                tp_plus_fp_cs = defaultdict(list)
                fp_os = defaultdict(list)
                tp_unk = defaultdict(list)
                fn_all = defaultdict(list)
                fn_k = defaultdict(list)
                thresh_AOSEs = defaultdict(list)
                recall_AOSEs = defaultdict(list)
                thresh_wis = defaultdict(list)
                confidence = defaultdict(list)
                tps = defaultdict(list)
                num_gt = defaultdict(list)
                difficult = defaultdict(list)
                
                eval_categories = self.known_classes + ['unknown']
                passed_cls_ids = []
                # for cls_idx, cls_name in enumerate(self._class_names): # num classes
                for cls_name in eval_categories: # num classes
                    cls_id = self.cat2id[cls_name]
                    cls_idx = self.id_map[cls_id]
                    passed_cls_ids.append(cls_id)

                    lines = predictions.get(cls_idx, [""])
            
                    if lines == [""]:
                        self._logger.info('{} / {}: "{}" has {} predictions. '.format(cls_idx, eval_categories.__len__(), cls_name, 0))
                    else:
                        self._logger.info('{} / {}: "{}" has {} predictions. '.format(cls_idx, eval_categories.__len__(), cls_name, len(lines)))

                    with open(res_file_template.format(cls_name), "w") as f:
                        f.write("\n".join(lines))

                    # for thresh in range(50, 100, 5):
                    thresh = 50 # consider detection if the IOU is larger than thresh
                    rec, prec, ap, unk_det_as_known, num_unk, tp_plus_fp_closed_set, fp_open_set,\
                    is_tp_unk, is_fn_all, is_difficult, is_fn_k, thresh_AOSE, recall_AOSE, thresh_wi, conf, tp, npos = \
                        voc_eval(
                        res_file_template, # '/tmp/pascal_voc_eval_txif1wu7/{}.txt'
                        # self._anno_file_template,# os.path.join(meta.dirname, "Annotations", "{}.xml")
                        # self._image_set_path,# os.path.join(meta.dirname, "ImageSets", "Main", meta.split + ".txt")
                        self._coco_api,
                        self.id2cat,
                        cls_name,
                        self.eval_index_range,
                        self.id_map,
                        ovthresh=thresh / 100.0,
                        use_07_metric=self._is_2007,
                        known_classes=self.known_classes
                    )
                    aps[thresh].append(ap * 100)
                    unk_det_as_knowns[thresh].append(unk_det_as_known)
                    num_unks[thresh].append(num_unk)
                    all_precs[thresh].append(prec)
                    all_recs[thresh].append(rec)
                    tp_plus_fp_cs[thresh].append(tp_plus_fp_closed_set)
                    fp_os[thresh].append(fp_open_set)
                    tp_unk[thresh].append(is_tp_unk)
                    fn_all[thresh].append(is_fn_all)
                    difficult[thresh].append(is_difficult)
                    fn_k[thresh].append(is_fn_k)
                    thresh_AOSEs[thresh].append(thresh_AOSE)
                    recall_AOSEs[thresh].append(recall_AOSE)
                    thresh_wis[thresh].append(thresh_wi)
                    confidence[thresh].append(conf)
                    tps[thresh].append(tp)
                    num_gt[thresh].append(npos)
                    
                    try:
                        recs[thresh].append(rec[-1] * 100)
                        precs[thresh].append(prec[-1] * 100) # 各カテゴリ，recall最大時のprecisionとrecallを保存
                    except:
                        recs[thresh].append(0)
                        precs[thresh].append(0)

            
            def get_recall_conf(confidence, num_gt, tps):
                # calc overall recall
                all_confidences = np.concatenate(confidence[50][:-1])
                num_gt = np.array(num_gt[50][:-1])
                num_gt = num_gt.sum()
                conf_sort_ind = all_confidences.argsort()[::-1]
                all_confidences = all_confidences[conf_sort_ind]
                all_tps = np.concatenate(tps[50][:-1])[conf_sort_ind]
                all_tps = all_tps.cumsum()
                entire_recall = all_tps / num_gt
                
                recall_conf = {} 
                for r in np.arange(0, 1, 0.001):
                    r = round(r, 2)
                    if all_confidences.__len__() == 0:
                        recall_conf[r] = -1
                        continue
                    else:
                        ind = np.abs(entire_recall - r).argmin()    
                        recall_conf[r] = all_confidences[ind]
                return recall_conf
            
            recall_conf = get_recall_conf(confidence, num_gt, tps)
            import pickle
            with open(os.path.join(split_dir, 'recall_conf.pkl'), 'wb') as f:
                pickle.dump(recall_conf, f)
                            
            if num_unks[thresh][-1] > 0:
                # get udr/udp/ur
                udr = {}
                udp = {}
                ur = {}
                # integrate unknown's FalsePositive (i.e., de tected as known) in each image  
                for thresh in tp_unk.keys():
                    # remove GT with difficult tag
                    tp_unk[thresh][-1] = tp_unk[thresh][-1][difficult[thresh][-1]==False]
                    fn_all[thresh][-1] = fn_all[thresh][-1][difficult[thresh][-1]==False]
                    for i in range(fn_k[thresh].__len__()-1):
                        fn_k[thresh][i] = fn_k[thresh][i][difficult[thresh][-1]==False] # remove difficult 
                        fn_k[thresh][i] *= fn_all[thresh][-1] # confirm fn* is subset of fn
                        if fn_k[thresh][-1] is None:
                            fn_k[thresh][-1] = fn_k[thresh][i]
                            continue
                        fn_k[thresh][-1] = np.maximum(fn_k[thresh][-1], fn_k[thresh][i])
                                    
                    unk_tp = tp_unk[thresh][-1].sum()
                    unk_fn = fn_all[thresh][-1].sum()
                    unk_GT = num_unks[thresh][-1].sum()
                    assert unk_GT == (unk_tp + unk_fn)
                    unk_fn_k = fn_k[thresh][-1].sum()
                    
                    udr[thresh] = (unk_tp + unk_fn_k) / (unk_tp + unk_fn)
                    udp[thresh] = unk_tp / (unk_tp + unk_fn_k)
                    ur[thresh] = unk_tp / (unk_tp + unk_fn)
                    
                self._logger.info('Unknown-Detection-Recall (iou={}): {}'.format(thresh, udr[thresh]))
                self._logger.info('Unknown-Detection-Precision (iou={}): {}'.format(thresh, udp[thresh]))
                self._logger.info('Unknown-Recall (iou={}): {}'.format(thresh, ur[thresh]))
            
            wi = self.compute_WI_at_many_recall_level(all_recs, tp_plus_fp_cs, fp_os, passed_cls_ids)
            self._logger.info('Wilderness Impact: \n{}'.format(str(wi)))
            
            # not average.
            precision_unk_at_each_recall = self.compute_avg_precision_at_many_recall_level_for_unk(all_precs, all_recs, passed_cls_ids) # Precision at recall=[0.1, 0.2, ..., 0.9] for [unknown]カテゴリ
            self._logger.info('Precision for only unknown class @ recalls: \n{}'.format(str(precision_unk_at_each_recall))) 

            ret = OrderedDict()

            mAP = {iou: np.mean(np.array(x)[ ~np.isnan(np.array(x)) ]) for iou, x in aps.items()} # mAP of knowns + unknown category
            ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP{}".format(thresh): mAP[thresh]}

            total_num_unk_det_as_known = {iou: np.sum(x) for iou, x in unk_det_as_knowns.items()} # num of A-OSE for all categories
            total_num_unk = num_unks[50][0] # assert num_unks[50][0] == num_unks[50][1] == ... == num_unks[50][num_cat]
            self._logger.info('Absolute OSE : ' + str(total_num_unk_det_as_known))
            self._logger.info('total_num_unk GT: ' + str(total_num_unk))
            
            from pathlib import Path
            import matplotlib.pyplot as plt
            thresh_AOSEs = np.array(thresh_AOSEs[50][:-1]).sum(0) # sum of all category
            plt.bar(np.arange(thresh_AOSEs.shape[0]), thresh_AOSEs)
            plt.savefig(os.path.join(Path(paper_measure_save_path).parent, 'thresh_AOSE.png'))
        
            recall_AOSEs = np.array(recall_AOSEs[50][:-1]).sum(0) # sum of all category
            plt.bar(np.arange(recall_AOSEs.shape[0]), recall_AOSEs)
            plt.savefig(os.path.join(Path(paper_measure_save_path).parent, 'recall_AOSE.png'))
            
            thresh_wis = np.array(thresh_wis[50][:-1])
            thresh_wis = thresh_wis[thresh_wis!=-1].reshape(-1, thresh_wis.shape[-1])
            thresh_wis = thresh_wis.mean(0)
            plt.bar(np.arange(thresh_wis.shape[0]), thresh_wis)
            plt.savefig(os.path.join(Path(paper_measure_save_path).parent, 'thresh_WI.png'))
            
            # Extra logging of class-wise APs
            avg_precs = list(np.mean([x for _, x in aps.items()], axis=0)) # mean per IOU
            
            for idx, name in enumerate(eval_categories):
                self._logger.info('AP(iou={}) for {}: {}'.format(thresh, name, aps[thresh][idx]))

            prev_ap = np.array(aps[thresh])[:self.prev_intro_cls]
            if self.prev_intro_cls > 0:
                if prev_ap.__len__() == 0:
                    prev_ap = np.array([0])
                self._logger.info("Prev known mAP(iou={}): {}".format(thresh, str(prev_ap[~np.isnan(prev_ap)].mean())))

            cur_ap = np.array(aps[thresh])[self.prev_intro_cls:self.prev_intro_cls + self.curr_intro_cls]
            if cur_ap.__len__() == 0:
                cur_ap = np.array([0])
            self._logger.info("Current known mAP(iou={}): {}".format(thresh, str(cur_ap[~np.isnan(cur_ap)].mean())))

            both_ap = np.array(aps[thresh])[:self.prev_intro_cls + self.curr_intro_cls]
            if both_ap.__len__() == 0:
                both_ap = np.array([0])
            self._logger.info("Both mAP(iou={}): {}".format(thresh, str(both_ap[~np.isnan(both_ap)].mean())))

            # self._logger.info("\nUnknown AP__: " + str(avg_precs[-1]))
            # self._logger.info("Unknown Precisions(iou=50) @ highest recall: " + str(precs[50][-1]))
            # self._logger.info("Unknown Recall(iou=50) @ highest recall: " + str(recs[50][-1]))
            self._logger.info("Unknown mAP(iou={}): {}".format(thresh, str(aps[thresh][-1])))
            
            import codecs
            paper_measure = [
                'WI @ recall==0.8: {}'.format(wi[0.8][50]), 
                'A-OSE: {}'.format(total_num_unk_det_as_known[50]), 
                'mAP for Previous Known: {}'.format(str(prev_ap[~np.isnan(prev_ap)].mean())),
                'mAP for Current Known: {}'.format(str(cur_ap[~np.isnan(cur_ap)].mean())),
                'mAP for Both: {}'.format(str(both_ap[~np.isnan(both_ap)].mean())),
                'mAP for only Unknown: {}'.format(aps[50][-1]),
                'thresh AOSE: {}'.format(thresh_AOSEs.tolist()),
                'recall AOSE: {}'.format(recall_AOSEs.tolist()),
                'WI @ recall=all: {}'.format([wi[rec][50] for rec in wi.keys()]),
                'WI @ mean: {}'.format(
                    (np.array([wi[rec][50] for rec in wi.keys()]) * 0.05).sum()
                ),
                'thresh WI: {}'.format(thresh_wis.tolist())
            ]
            if num_unks[thresh][-1] > 0:
                paper_measure.extend(
                    [
                        'Unknown-Recall: {}'.format(ur[50]),
                        'Unknown-Detection-Recall: {}'.format(udr[50]),
                        'Unknown-Detection-Precision: {}'.format(udp[50]),
                    ]
                )
            rhandler = get_api_result_handler(paper_measure_save_path)
            self._logger.addHandler(rhandler)
            self._logger.info('\n'.join(paper_measure))
            _ = self._logger.handlers.pop(-1)
        
        # back-convert [unknown] -> [original category]
        # self._coco_api.anns = org_gt_anns
        self._coco_api.dataset['annotations'] = list(org_gt_anns.values())
        self._coco_api.createIndex()
        
        return ret
     

"""Python implementation of the PASCAL VOC devkit's AP evaluation code."""

def parse_rec(anns, id2cat, known_classes, eval_index_range, id_map):
    """
    reconstruct gt data : if GT are in unknown category, change them into "unknown"
    """
    
    # Parse a PASCAL VOC xml file
    VOC_CLASS_NAMES_COCOFIED = [
        "airplane", "dining table", "motorcycle",
        "potted plant", "couch", "tv"
    ]
    BASE_VOC_CLASS_NAMES = [
        "aeroplane", "diningtable", "motorbike",
        "pottedplant", "sofa", "tvmonitor"
    ]
    objects = []
    for an in anns:
        obj_struct = {}
        
        cls_name = id2cat[an['category_id']]

        # [already done the same process in convert_to_unknown at COCO perfomance]
        # if cls_name not in known_classes:
        #     cls_name = 'unknown' # unknown but learned in downstream task
        #     if id_map[an['category_id']] not in eval_index_range: # index is not in range
        #         an['iscrowd'] = True # to ignore this GT in evaluation

        # valid only for COCO dataset
        if cls_name in VOC_CLASS_NAMES_COCOFIED:
            cls_name = BASE_VOC_CLASS_NAMES[VOC_CLASS_NAMES_COCOFIED.index(cls_name)]

        obj_struct["name"] = cls_name
        obj_struct['difficult'] = int(an['iscrowd']) # use iscrowd in place of difficult
        bbox = an['bbox']
        obj_struct["bbox"] = [
            int(bbox[0]), # x1
            int(bbox[1]), # y1
            int(bbox[0]+bbox[2]), # x1+w
            int(bbox[1]+bbox[3],) # y1+h
        ]
        objects.append(obj_struct)
    return objects


def voc_AOSE(rec, is_unk):
    """AOSE-Recall
    """
    assert rec.shape == is_unk.shape
    # is_unk = is_unk.cumsum() / is_unk.sum()
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], rec, [1.0]))
    munk = np.concatenate(([0.0], is_unk.cumsum(), [is_unk.sum()]))

    # # compute the precision envelope
    # for i in range(mpre.size - 1, 0, -1):
    #     mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
    # for i in range(mrec.size - 1, 0, -1):
    for i in range(0, mrec.size-1, 1):
        munk[i+1] =  np.maximum(munk[i], munk[i+1])     
        # mrec[i+1] =  np.maximum(mrec[i], mrec[i+1])        
        
    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
    # i = np.where(munk[1:] != munk[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * munk[i + 1])
    # ap = np.sum((munk[i + 1] - munk[i]) * mrec[i + 1])
    
    return ap


def avg_AOSE(is_unk, confidence):
    """scorethrshold - Recall
    """
    cumsum_unk = np.cumsum(is_unk)
    
    mconf = np.concatenate(([0.0], confidence, [1.0]))
    munk = np.concatenate(([0.0], cumsum_unk, [is_unk.sum()]))
    
    score_idx = np.where(confidence[1:] != confidence[:-1])[0]
    average_AOSE = np.sum((confidence[score_idx] - confidence[score_idx + 1]) * cumsum_unk[score_idx + 1])        
    return average_AOSE


def recall_AOSE_func(rec, is_unk):
    '''get AOSE value in each recall level
    '''
    assert rec.shape == is_unk.shape
    if rec.__len__() == is_unk.__len__() == 0:
            return np.zeros_like(np.arange(0.05, 1, 0.05))
        
    cumsum_unk = np.cumsum(is_unk)
    AOSE = {}
    for r in np.arange(0.05, 1, 0.05): # [0.05 - 0.95]
        # find nearlest recall index
        r = round(r, 2)
        r_ind = np.abs(rec - r).argmin()
        AOSE[r] = cumsum_unk[r_ind]
    # return AOSE
    return np.array(list(AOSE.values()))


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def voc_eval(detpath, coco_api, id2cat, classname, eval_index_range, id_map, ovthresh=0.5, use_07_metric=False, known_classes=None):

    # === 
    imagenames = coco_api.getImgIds()
    imagenames_filtered = []
    # load annots
    recs = {}
    num_k_inst = 0
    num_unk_inst = 0 # num of unknown

    for imagename in imagenames:
        # rec = parse_rec(annopath.format(imagename), tuple(known_classes))
        # Convert GT's category into Unknown, if the category is not contained in Known
        # anns = coco_api.loadAnns(ids=coco_api.getAnnIds(imgIds=[imagename]))
        ann_ids = coco_api.getAnnIds(imgIds=[imagename])
        anns = []
        for aid in ann_ids:         
            try:
                ann = coco_api.loadAnns(ids=[aid])[0]
                anns.append(ann)
            except KeyError: # to prevent removing all gts in image because of KeyError (hope to get known gts)
                continue 
        
        rec = parse_rec(
            anns,
            id2cat, 
            known_classes,
            eval_index_range,
            id_map,
        )
        if rec is not None:
            for r in rec:
                if r['name'] == 'unknown':
                    num_unk_inst += 1
                else:
                    num_k_inst += 1
            recs[imagename] = rec # store GTs
            imagenames_filtered.append(imagename)
    # print('num known GT instances @ test time: {}'.format(num_k_inst))
    # print('num unknown GT instances @ test time: {}'.format(num_unk_inst))
    imagenames = imagenames_filtered

    # extract gt objects for this class
    class_recs = {}
    npos = 0 
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        # difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool_)
        det = np.array([False] * len(R)).astype(np.int64)
        npos = npos + sum(~difficult)  # num of GT for this category
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}
        
    if classname != 'unknown':
            print('num gts for {}: {}/{}'.format(classname, npos, num_k_inst))
    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids) # num of predictions as this category
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    
    for d in range(nd): # detection
        R = class_recs[image_ids[d]] # gt
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1 # flag for already used GT
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0
    
    if classname == 'unknown':
        is_tp_unk = np.hstack([R['det'] for imagename, R in class_recs.items() if R['bbox'].size > 0])
        is_fn = (~(np.hstack([R['det'] for imagename, R in class_recs.items() if R['bbox'].size > 0]).astype(np.bool_))).astype(np.int64)
        is_difficult = np.hstack([R['difficult'] for imagename, R in class_recs.items() if R['bbox'].size > 0])

    # compute precision recall
    fp = np.cumsum(fp) # 
    org_tp = tp.copy()
    tp = np.cumsum(tp) # TP for this category
    rec = tp / float(npos)
    
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    # plot_pr_curve(prec, rec, classname+'.png')
    ap = voc_ap(rec, prec, use_07_metric)
    '''
    Computing Absolute Open-Set Error (A-OSE) and Wilderness Impact (WI)
                                    ===========    
    Absolute OSE = # of unknown objects classified as known objects of class 'classname'
    WI = FP_openset / (TP_closed_set + FP_closed_set)
    '''
    logger = logging.getLogger(__name__)

    # Finding GT of unknown objects
    unknown_class_recs = {}
    n_unk = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == 'unknown']
        bbox = np.array([x["bbox"] for x in R])
        # difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool_)
        det = [False] * len(R)
        n_unk = n_unk + sum(~difficult) # exclude "difficult" GT from counting
        det_as_known = (np.array([False] * len(R))*1).astype(np.int64)
        unknown_class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det, 'det_as_known': det_as_known,}
                
    if classname == 'unknown':        
        return rec, prec, ap, 0, n_unk, None, None, is_tp_unk, is_fn, is_difficult,\
               None, None, None, None, None, None, None # tp+fp, np.cumsum(fp) ?
        # return rec, prec, ap, 0, n_unk, tp+fp, None, None # tp+fp, np.cumsum(fp) ?
    
    # Go down each detection and see if it has an overlap with an unknown object.
    # If so, it is an unknown object that was classified as known.
    is_unk = np.zeros(nd)
    for d in range(nd): # for all predictions in this image
        R = unknown_class_recs[image_ids[d]] # unknown GTs of this image
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0: # if there is unknown GT
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
        
        if ovmax > ovthresh and (not R["difficult"][jmax]):
            is_unk[d] = 1.0 # flag for detection
            R['det_as_known'][jmax] = 1 # flag for GT
    
    import matplotlib.pyplot as plt
    # confidence = confidence[sorted_ind]
    # avgAOSE = self.avg_AOSE(is_unk, confidence) # AUC of score_thresh - AOSE
    # vocAOSE = self.voc_AOSE(rec, is_unk) # AUC of recall - AOSE
    
    # get AOSE per threshold (bins)        
    add = 0.05
    bins = []
    confidence = confidence[sorted_ind]
    for th in np.arange(0, 1, add): # [0, 0.05, 0.10, .., 0.95] 20
        th = round(th, 2)
        i = np.where(np.logical_and(confidence > th, confidence <= th+add))[0]
        bins.append(is_unk[i].sum())
                                
    thresh_AOSE = np.array(bins) # AOSE along with confidence [0, 0.05, 0.10, ..., 0.90, 0.95] = (20)
    recall_AOSE = recall_AOSE_func(rec, is_unk) # AOSE along with recall [0.05, 0.10, ..., 0.90, 0.95] = (19)
    
    is_unk_sum = np.sum(is_unk)       
    is_fn_k = np.hstack([R['det_as_known'] for imagename, R in unknown_class_recs.items() if R['bbox'].size > 0])        
    tp_plus_fp_closed_set = tp+fp
    fp_open_set = np.cumsum(is_unk) # known prediction (fp) as unknown GT = false positive
         
    wi = fp_open_set / tp_plus_fp_closed_set
    thresh_wi = []
    for th in np.arange(0, 1, add): # [0, 0.05, 0.10, ..., 0.90, 0.95] = (20)
        th = round(th, 2)
        if len(wi)==0:
            thresh_wi.append(-1)
            continue
        else:
            conf_ind = np.abs(confidence - th).argmin()
        thresh_wi.append(wi[conf_ind]) # WI along with confidence

    '''
    rec: recall on this category (classname)
    prec: precision on this category (classname)
    ap: average precision on this category (classname)
    is_unk_sum: num of predictions as "classname" category which is mistakenly detected as "unknown" GTs
    n_unk: num of unknown GTs
    tp_plus_fp_closed_set:  tp + fp (fp contains "overlap with another known category GTs" and "overlap with no GTs", not contains "fp on unknown GTs")
    fp_open_set: fp on unknown GTs
    fn_as_known: false negatives(i.e., num of GTs) for unknown GTs that detected as "classname" in each image
    '''
    return rec, prec, ap, is_unk_sum, n_unk, tp_plus_fp_closed_set, fp_open_set, None, None, None,\
           is_fn_k, thresh_AOSE, recall_AOSE, thresh_wi, confidence, org_tp, npos


def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    if hasattr(instances, 'fg_bg_scores'):
        fg_bg_scores = instances.fg_bg_scores
        fg_bg_scores = fg_bg_scores.tolist()
    else:
        fg_bg_scores = [[] for _ in range(classes.__len__())]
    
    if hasattr(instances, 'embed'):
        embed = instances.embed
        embed = embed.tolist()
    else:
        embed = [[] for _ in range(classes.__len__())]
    
    if hasattr(instances, 'all_scores'):
        all_scores = instances.all_scores
        v, ind = all_scores.topk(3)
        top1, top2 = v[:,0], v[:, 1]
        score_ratio = (top1/top2).tolist()
    else:
        score_ratio = [[] for _ in range(classes.__len__())]

    if hasattr(instances, 'sigmoid_scores') and hasattr(instances, 'sigmoid_label_index'):
        sigmoid_scores = instances.sigmoid_scores.tolist()
        sigmoid_label_index = instances.sigmoid_label_index.tolist()        

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    results = []
    assert num_instance == classes.__len__()
    for k in range(num_instance):
        if classes[k] == -100:
            continue 
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
            "ratio": score_ratio[k],
            "fg_bg_scores": fg_bg_scores[k],
            "embed": embed[k],
            "sigmoid_scores": sigmoid_scores[k] if hasattr(instances, 'sigmoid_scores') else None,
            "sigmoid_label_index": sigmoid_label_index[k] if hasattr(instances, 'sigmoid_label_index') else None,  
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            # In COCO annotations,
            # keypoints coordinates are pixel indices.
            # However our predictions are floating point coordinates.
            # Therefore we subtract 0.5 to be consistent with the annotation format.
            # This is the inverse of data loading logic in `datasets/coco.py`.
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        results.append(result)
    return results


# inspired from Detectron:
# https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L255 # noqa
def _evaluate_box_proposals(dataset_predictions, coco_api, thresholds=None, area="all", limit=None):
    """
    Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for prediction_dict in dataset_predictions:
        predictions = prediction_dict["proposals"]

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = predictions.objectness_logits.sort(descending=True)[1]
        predictions = predictions[inds]

        ann_ids = coco_api.getAnnIds(imgIds=prediction_dict["image_id"])
        anno = coco_api.loadAnns(ann_ids)
        gt_boxes = [
            BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            for obj in anno
            if obj["iscrowd"] == 0
        ]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = Boxes(gt_boxes)
        gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

        if len(gt_boxes) == 0 or len(predictions) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if limit is not None and len(predictions) > limit:
            predictions = predictions[:limit]

        overlaps = pairwise_iou(predictions.proposal_boxes, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(predictions), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = (
        torch.cat(gt_overlaps, dim=0) if len(gt_overlaps) else torch.zeros(0, dtype=torch.float32)
    )
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }


