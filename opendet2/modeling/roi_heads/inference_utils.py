# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import logging
import math
import os
import random
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.distributions as dists
from torch.distributions import Categorical
from detectron2.config import configurable
from detectron2.layers import (ShapeSpec, batched_nms, cat, cross_entropy,
                               nonzero_tuple)
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import (FastRCNNOutputLayers, _log_classification_stats)
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.structures.boxes import matched_boxlist_iou
from detectron2.utils import comm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F
from pycocotools import mask as maskUtils
import random

from .nms_utils import sum_softmax_entropy

# =================================================================================================

def fast_rcnn_inference(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    embed: List[torch.Tensor],
    objectness: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float, # 0.05
    nms_thresh: float, # 0.5
    topk_per_image: int, # 100
    vis_iou_thr: float = 1.0,
    extracts_embed = False
):
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, embed_per_image, objectness_per_image, image_shape, 
            score_thresh, nms_thresh, topk_per_image, vis_iou_thr, extracts_embed = extracts_embed
        )
        for scores_per_image, boxes_per_image, embed_per_image, objectness_per_image, image_shape in zip(scores, boxes, embed, objectness, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image(
    boxes,
    scores,
    embed, 
    objectness,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    vis_iou_thr: float,
    extracts_embed,
):
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)

    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    fg_bg_scores = scores.clone() # [N, K+1] (not K+2; we don't expect that the OpenDet classifier uses this function)

    scores = scores[:, :-1] # remove bg class
    unk_label_ind = scores.shape[1] - 1 # K-bg last index == NUM_CLASSES
    
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero() # [index of ROI, class_id]
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    all_scores = scores[filter_inds[:, 0], :].clone()
    fg_bg_scores = fg_bg_scores[filter_inds[:, 0]]
    embed = embed[filter_inds[:, 0]]
    scores = scores[filter_mask]
    
    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    all_scores = all_scores[keep]
    fg_bg_scores = fg_bg_scores[keep]
    embed = embed[keep]

    assert all_scores.size(0) == scores.size(0)

    # apply nms between known classes and unknown class for visualization.
    if vis_iou_thr < 1.0:
        boxes, scores, filter_inds = unknown_aware_nms(boxes, scores, filter_inds, ukn_class_id=unk_label_ind, iou_thr=vis_iou_thr)

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.all_scores = all_scores
    result.fg_bg_scores = fg_bg_scores
    result.pred_classes = filter_inds[:, 1]
    if extracts_embed:
        result.embed = embed
    return result, filter_inds[:, 0]


# =======================================================================================================

def fast_rcnn_inference_sigmoid(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    embed: List[torch.Tensor],
    objectness: List[torch.Tensor], 
    image_shapes: List[Tuple[int, int]],
    score_thresh: float, # 0.05
    nms_thresh: float, # 0.5
    topk_per_image: int, # 100
    vis_iou_thr: float = 1.0,
    extracts_embed = False,
):
    result_per_image = [
        fast_rcnn_inference_single_image_sigmoid(
            boxes_per_image, scores_per_image, embed_per_image, objectness_per_image, image_shape,
            score_thresh, nms_thresh, topk_per_image, vis_iou_thr, extracts_embed
        )
        for scores_per_image, boxes_per_image, embed_per_image, objectness_per_image, image_shape in zip(scores, boxes, embed, objectness, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image_sigmoid(
    boxes,
    scores,
    embed, 
    objectness,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    vis_iou_thr: float,
    extracts_embed = False,
):
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)

    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    fg_bg_scores = scores.clone() # [N, K+1] (not K+2; we don't expect that the OpenDet classifier uses this function)

    # scores = scores[:, :-1] # all columns are valid classes in sigmoid classifier
    unk_label_ind = scores.shape[1] - 1 # K-bg last index == NUM_CLASSES
    
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero() # [index of ROI, class_id]
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    all_scores = scores[filter_inds[:, 0], :].clone()
    fg_bg_scores = fg_bg_scores[filter_inds[:, 0]]
    embed = embed[filter_inds[:, 0]]
    scores = scores[filter_mask]
    
    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    all_scores = all_scores[keep]
    fg_bg_scores = fg_bg_scores[keep]
    embed = embed[keep]

    assert all_scores.size(0) == scores.size(0)

    # apply nms between known classes and unknown class for visualization.
    if vis_iou_thr < 1.0:
        boxes, scores, filter_inds = unknown_aware_nms(boxes, scores, filter_inds, ukn_class_id=unk_label_ind, iou_thr=vis_iou_thr)

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.all_scores = all_scores
    result.fg_bg_scores = fg_bg_scores
    result.pred_classes = filter_inds[:, 1]
    if extracts_embed:
        result.embed = embed
    return result, filter_inds[:, 0]


# =======================================================================================================


def fast_rcnn_inference_MSP(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    embed: List[torch.Tensor],
    objectness: List[torch.Tensor], 
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    vis_iou_thr: float = 1.0,
    msp_th: float = -1.0,
    after_nms: bool = False,
    extracts_embed = False
):
    result_per_image = [
        fast_rcnn_inference_single_image_MSP(
            boxes_per_image, scores_per_image, embed_per_image, objectness_per_image, image_shape, score_thresh, nms_thresh, topk_per_image, vis_iou_thr,
            msp_th, after_nms, extracts_embed
        )
        for scores_per_image, boxes_per_image, embed_per_image, objectness_per_image, image_shape in zip(scores, boxes, embed, objectness, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image_MSP(
    boxes,
    scores,
    embed,
    objectness,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    vis_iou_thr: float,
    msp_th: float,
    after_nms: bool,
    extracts_embed,
):
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
        
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    fg_bg_scores = scores.clone() # [N, K+1] (not K+2; we don't expect that the OpenDet classifier uses this function)

    scores = scores[:, :-1] # remove bg class 
    unk_label_ind = scores.shape[1] # K-bg last index == NUM_CLASSES
    
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    all_scores = scores[filter_inds[:, 0], :].clone()
    fg_bg_scores = fg_bg_scores[filter_inds[:, 0], :]
    embed = embed[filter_inds[:, 0], :]
    scores = scores[filter_mask]
    
    # 1.5. Convert some predictions into unknown category, before NMS
    if not after_nms:
        scores, all_scores, filter_inds = unknown_detection_MSP(scores, all_scores, filter_inds, msp_th, unk_label_ind)

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh) # default: nms_thresh = 0.5
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    all_scores = all_scores[keep]
    fg_bg_scores = fg_bg_scores[keep]
    embed = embed[keep]

    assert all_scores.size(0) == scores.size(0)

    # 2.5. Convert some predictions into unknown category, after NMS
    if after_nms:
        scores, all_scores, filter_inds = unknown_detection_MSP(scores, all_scores, filter_inds, msp_th, unk_label_ind)
        # Apply NMS to remove duplicated unknown
        keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh) # default: nms_thresh = 0.5
        if topk_per_image >= 0:
            keep = keep[:topk_per_image]
        boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
        all_scores = all_scores[keep]
        fg_bg_scores = fg_bg_scores[keep]
        embed = embed[keep]
        sigmoid_logit_scores = [sls[keep] for sls in sigmoid_logit_scores]

    # apply nms between known classes and unknown class for visualization.
    if vis_iou_thr < 1.0:
        boxes, scores, filter_inds = unknown_aware_nms(boxes, scores, filter_inds, ukn_class_id=unk_label_ind, iou_thr=vis_iou_thr)

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.all_scores = all_scores
    result.fg_bg_scores = fg_bg_scores
    result.pred_classes = filter_inds[:, 1]
    if extracts_embed:
        result.embed = embed
    return result, filter_inds[:, 0]


def unknown_detection_MSP(scores, all_scores, filter_inds, msp_th, unk_label_ind):
    assert scores.size(0) == all_scores.size(0) == filter_inds.size(0)
    num_preds = scores.size(0)
    if num_preds == 0:
        return scores, all_scores, filter_inds
    
    for b in range(num_preds):
        sd = all_scores[b]

        # if scores[b] < msp_th: 
        if sd.max() < msp_th:
            # update confidence into unknown score
            # unk_score = torch.min(torch.ones(1).type_as(scores), value.sum())
            # scores[b] = unk_score
            # update cls label into unknown's
            filter_inds[b,1] = unk_label_ind
    return scores, all_scores, filter_inds

# ============================================================

def fast_rcnn_inference_ratio_ud(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    embed: List[torch.Tensor],
    objectness: List[torch.Tensor], 
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    vis_iou_thr: float = 1.0,
    ratio_ud_unk_ratio: float = -1.0,
    after_nms: bool = False, # if True, conduct class-wise NMS after unknown conversion
    compare_bg: bool = False, # if True, take into account the BG scores when to convert known into unknown
    bg_to_unk: bool = False, # if True, remove BG boxes 
    inference_wo_bg: bool = False,
    keep_bg: bool = False,
    sigmoid_logit_scores = None,
    extracts_embed = False,
):
    # if inference_wo_bg:
    #     result_per_image = [
    #         fast_rcnn_inference_single_image_ratio_ud_wo_bg(
    #             boxes_per_image, scores_per_image, objectness_per_image, image_shape, score_thresh, nms_thresh, topk_per_image, vis_iou_thr,
    #             ratio_ud_unk_ratio, after_nms
    #         )
    #         for scores_per_image, boxes_per_image, objectness_per_image, image_shape in zip(scores, boxes, objectness, image_shapes)
    #     ]
    # elif keep_bg:
    #     result_per_image = [
    #         fast_rcnn_inference_single_image_ratio_ud_keep_bg(
    #             boxes_per_image, scores_per_image, objectness_per_image, image_shape, score_thresh, nms_thresh, topk_per_image, vis_iou_thr,
    #             ratio_ud_unk_ratio, after_nms, compare_bg
    #         )
    #         for scores_per_image, boxes_per_image, objectness_per_image, image_shape in zip(scores, boxes, objectness, image_shapes)
    #     ]
    # else:
    result_per_image = [
        fast_rcnn_inference_single_image_ratio_ud(
            boxes_per_image, scores_per_image, embed_per_image, objectness_per_image, image_shape, score_thresh, nms_thresh, topk_per_image, vis_iou_thr,
            ratio_ud_unk_ratio, after_nms, compare_bg, bg_to_unk, sigmoid_logit_scores=sigmoid_logit_scores, extracts_embed=extracts_embed,
        )
        for scores_per_image, boxes_per_image, embed_per_image, objectness_per_image, image_shape in zip(scores, boxes, embed, objectness, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image_ratio_ud(
    boxes,
    scores,
    embed,
    objectness,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    vis_iou_thr: float,
    ratio_ud_unk_ratio: float,
    after_nms: bool,
    compare_bg: bool = False,
    bg_to_unk: bool = False,
    sigmoid_logit_scores = None,
    extracts_embed = False,
):
    if sigmoid_logit_scores is None:
        sigmoid_logit_scores = [-torch.ones_like(scores)]
    
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
        
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        sigmoid_logit_scores = [sls[valid_mask] for sls in sigmoid_logit_scores]
    
    fg_bg_scores = scores.clone() # [N, K+1] (not K+2; we don't expect that the OpenDet classifier uses this function)

    if bg_to_unk:
        # disable bg removal
        # scores = scores[:, :-1] 
        unk_label_ind = scores.shape[1]-1 # K-bg last index == NUM_CLASSES
        bg_label_ind = scores.shape[1]
    else:
        scores = scores[:, :-1] # remove bg class 
        unk_label_ind = scores.shape[1] # K-bg last index == NUM_CLASSES
    
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    
     # when comparing the confidence ratios, we consider background confidence, not only known classes's.
    if compare_bg:
        all_scores = fg_bg_scores[filter_inds[:, 0], :].clone() # [N, K+bg]
    else:  
        all_scores = scores[filter_inds[:, 0], :].clone() # [N, K]
    
    fg_bg_scores = fg_bg_scores[filter_inds[:, 0], :]
    embed = embed[filter_inds[:, 0], :]
    scores = scores[filter_mask]
    sigmoid_logit_scores = [sls[filter_inds[:, 0]] for sls in sigmoid_logit_scores]
    
    # transform bg_label_ind-1 into bg_label_ind 
    if bg_to_unk:
        bg_mask = filter_inds[:, 1] == bg_label_ind - 1
        filter_inds[bg_mask, 1] = bg_label_ind
    
    # 1.5. Convert some predictions into unknown category, before NMS
    if not after_nms:
        scores, all_scores, filter_inds = convert_known_into_unknown(scores, all_scores, filter_inds, ratio_ud_unk_ratio, unk_label_ind)

    # remove all BG predictions
    if bg_to_unk:
        remove = (filter_inds[:,1] == bg_label_ind)
        scores = scores[~remove]
        all_scores = all_scores[~remove]
        boxes = boxes[~remove]
        filter_inds = filter_inds[~remove]
        fg_bg_scores = fg_bg_scores[~remove]
        embed = embed[~remove]
        sigmoid_logit_scores = [sls[~remove] for sls in sigmoid_logit_scores]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh) # default: nms_thresh = 0.5
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    all_scores = all_scores[keep]
    fg_bg_scores = fg_bg_scores[keep]
    embed = embed[keep]
    sigmoid_logit_scores = [sls[keep] for sls in sigmoid_logit_scores]

    assert all_scores.size(0) == scores.size(0)

    # 2.5. Convert some predictions into unknown category, after NMS
    if after_nms:
        scores, all_scores, filter_inds = convert_known_into_unknown(scores, all_scores, filter_inds, ratio_ud_unk_ratio, unk_label_ind)
        # Apply NMS to remove duplicated unknown
        keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh) # default: nms_thresh = 0.5
        if topk_per_image >= 0:
            keep = keep[:topk_per_image]
        boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
        all_scores = all_scores[keep]
        fg_bg_scores = fg_bg_scores[keep]
        embed = embed[keep]
        sigmoid_logit_scores = [sls[keep] for sls in sigmoid_logit_scores]
        
    # apply nms between known classes and unknown class for visualization.
    if vis_iou_thr < 1.0:
        boxes, scores, filter_inds = unknown_aware_nms(boxes, scores, filter_inds, ukn_class_id=unk_label_ind, iou_thr=vis_iou_thr)

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.all_scores = all_scores
    result.fg_bg_scores = fg_bg_scores
    result.pred_classes = filter_inds[:, 1]
    if sigmoid_logit_scores[0].mean() != -1: # if sigmoid_logit_scores was None, all elements should be -1.
        sig_scores, sig_labels = calc_sigmoid_label_scores(sigmoid_logit_scores)
        result.sigmoid_scores = sig_scores
        result.sigmoid_label_index = sig_labels
    if extracts_embed:
        result.embed = embed
    return result, filter_inds[:, 0]



def fast_rcnn_inference_single_image_ratio_ud_keep_bg(
    boxes,
    scores,
    objectness,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    vis_iou_thr: float,
    ratio_ud_unk_ratio: float,
    after_nms: bool,
    compare_bg: bool = False,
):
    '''keep background bounding boxes to make analysis on the following evaluation process.
    Thus, the mAP scores calculated by unsing this function will not be considered as proper results.
    '''    
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
        
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
    
    fg_bg_scores = scores.clone() # [N, K+1] (not K+2; we don't expect that the OpenDet classifier uses this function)

    unk_label_ind = scores.shape[1]-1 # K-bg last index == NUM_CLASSES
    bg_label_ind = scores.shape[1]
    
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    
     # when comparing the confidence ratios, we consider background confidence, not only known classes's.
    if compare_bg:
        all_scores = fg_bg_scores[filter_inds[:, 0], :].clone() # [N, K+bg]
    else:  
        all_scores = scores[filter_inds[:, 0], :].clone() # [N, K]
    
    bg_mask = (filter_inds[:, 1] == bg_label_ind-1)
    filter_inds[bg_mask, 1] = bg_label_ind # 0-k 
    
    fg_bg_scores = fg_bg_scores[filter_inds[:, 0], :]
    scores = scores[filter_mask]   
    
    # 1.5. Convert some predictions into unknown category, before NMS
    if not after_nms: # apply to only non-BG bboxes
        scores[~bg_mask], all_scores[~bg_mask], filter_inds[~bg_mask] = convert_known_into_unknown(
            scores[~bg_mask], all_scores[~bg_mask], filter_inds[~bg_mask], ratio_ud_unk_ratio, unk_label_ind
        )

    # escape NMS?
    
    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh) # default: nms_thresh = 0.5
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    all_scores = all_scores[keep]
    fg_bg_scores = fg_bg_scores[keep]

    assert all_scores.size(0) == scores.size(0)

    # 2.5. Convert some predictions into unknown category, after NMS
    if after_nms:
        scores, all_scores, filter_inds = convert_known_into_unknown(scores, all_scores, filter_inds, ratio_ud_unk_ratio, unk_label_ind)

    # apply nms between known classes and unknown class for visualization.
    if vis_iou_thr < 1.0:
        boxes, scores, filter_inds = unknown_aware_nms(boxes, scores, filter_inds, ukn_class_id=unk_label_ind, iou_thr=vis_iou_thr)

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.all_scores = all_scores
    result.fg_bg_scores = fg_bg_scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


def fast_rcnn_inference_single_image_ratio_ud_wo_bg(
    boxes,
    scores,
    objectness,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    vis_iou_thr: float,
    ratio_ud_unk_ratio: float,
    after_nms: bool,
    # compare_bg: bool = False,
    # bg_to_unk: bool = False,
):
    '''
    same as fast_rcnn_inference_single_image_ratio_ud but
    scores (after softmax scores) do not contain bg class,
    i.e., when function of logit -> softmax, we remove bg logits 
    '''
    
    # decay scores
    # scores *= objectness.view(-1, 1) 
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)

    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
    
    unk_label_ind = scores.shape[1]
    
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    
    all_scores = scores[filter_inds[:, 0], :].clone() # [N, K]
    scores = scores[filter_mask]
        
    # 1.5. Convert some predictions into unknown category, before NMS
    if not after_nms:
        scores, all_scores, filter_inds = convert_known_into_unknown(scores, all_scores, filter_inds, ratio_ud_unk_ratio, unk_label_ind)

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh) # default: nms_thresh = 0.5
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    all_scores = all_scores[keep]

    assert all_scores.size(0) == scores.size(0)

    # 2.5. Convert some predictions into unknown category, after NMS
    if after_nms:
        scores, all_scores, filter_inds = convert_known_into_unknown(scores, all_scores, filter_inds, ratio_ud_unk_ratio, unk_label_ind)

    # apply nms between known classes and unknown class for visualization.
    if vis_iou_thr < 1.0:
        boxes, scores, filter_inds = unknown_aware_nms(boxes, scores, filter_inds, ukn_class_id=unk_label_ind, iou_thr=vis_iou_thr)

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.all_scores = all_scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


def convert_known_into_unknown(scores, all_scores, filter_inds, unk_ratio_thresh, unk_label_ind):
    assert scores.size(0) == all_scores.size(0) == filter_inds.size(0)
    num_preds = scores.size(0)
    if num_preds == 0:
        return scores, all_scores, filter_inds
    
    for b in range(num_preds):
        value, _ = all_scores[b].topk(3)
        top1, top2 = value[:2]
        ratio_cond = ((top1/top2) < unk_ratio_thresh)
        
        if ratio_cond: 
            # 1. update confidence into unknown score
            # unk_score = torch.min(torch.ones(1).type_as(scores), value.sum())
            # 2. allow to be unk_score > 1.0
            unk_score = value.sum()
            scores[b] = unk_score
            # update cls label into unknown's
            filter_inds[b,1] = unk_label_ind
    return scores, all_scores, filter_inds

# =================================================================================================

def fast_rcnn_inference_entropy(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    embed: List[torch.Tensor],
    objectness: List[torch.Tensor], 
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    vis_iou_thr: float = 1.0,
    entropy_threshold: float = -1.0,
    after_nms: bool = False, # if True, conduct class-wise NMS after unknown conversion
    compare_bg: bool = False, # if True, take into account the BG scores when to convert known into unknown
    bg_to_unk: bool = False, # if True, remove BG boxes 
    inference_wo_bg: bool = False,
    sigmoid_logit_scores = None,
    extracts_embed = False,
):
    result_per_image = [
        fast_rcnn_inference_single_image_entropy(
        # fast_rcnn_inference_single_image_entropy_summation(
            boxes_per_image, scores_per_image, embed_per_image, objectness_per_image, image_shape, score_thresh, nms_thresh, topk_per_image, vis_iou_thr,
            entropy_threshold, after_nms, compare_bg, bg_to_unk,
            sigmoid_logit_scores = sigmoid_logit_scores, 
            extracts_embed = extracts_embed,
        )
        for scores_per_image, boxes_per_image, embed_per_image, objectness_per_image, image_shape in zip(scores, boxes, embed, objectness, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image_entropy(
    boxes,
    scores,
    embed,
    objectness,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    vis_iou_thr: float,
    entropy_threshold: float,
    after_nms: bool,
    compare_bg: bool = False,
    bg_to_unk: bool = False,
    sigmoid_logit_scores = None,
    extracts_embed = False,
):
    if sigmoid_logit_scores is None:
        sigmoid_logit_scores = [-torch.ones_like(scores)]
        
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        sigmoid_logit_scores = [sls[valid_mask] for sls in sigmoid_logit_scores]
    
    fg_bg_scores = scores.clone() # [N, K+1] (not K+2; we don't expect that the OpenDet classifier uses this function)

    if bg_to_unk:
        # disable bg removal
        # scores = scores[:, :-1] 
        unk_label_ind = scores.shape[1]-1 # K-bg last index == NUM_CLASSES
        bg_label_ind = scores.shape[1]
    else:
        scores = scores[:, :-1] # remove bg class 
        unk_label_ind = scores.shape[1] # K-bg last index == NUM_CLASSES
    
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    
     # when comparing the confidence ratios, we consider background confidence, not only known classes's.
    if compare_bg:
        all_scores = fg_bg_scores[filter_inds[:, 0], :].clone() # [N, K+bg]
    else:  
        all_scores = scores[filter_inds[:, 0], :].clone() # [N, K]
    
    fg_bg_scores = fg_bg_scores[filter_inds[:, 0], :]
    embed = embed[filter_inds[:, 0], :]
    scores = scores[filter_mask]
    sigmoid_logit_scores = [sls[filter_inds[:, 0]] for sls in sigmoid_logit_scores]
    
    # transform bg_label_ind-1 into bg_label_ind 
    if bg_to_unk:
        bg_mask = filter_inds[:, 1] == bg_label_ind - 1
        filter_inds[bg_mask, 1] = bg_label_ind
  
    entropies = Categorical(probs=all_scores).entropy()
  
    # 1.5. Convert some predictions into unknown category, before NMS
    if not after_nms:
        scores, filter_inds = convert_known_into_unknown_entropy(scores, filter_inds, entropies, entropy_threshold, unk_label_ind)
  
    # remove all BG predictions
    if bg_to_unk:
        remove = (filter_inds[:,1] == bg_label_ind)
        scores = scores[~remove]
        all_scores = all_scores[~remove]
        boxes = boxes[~remove]
        filter_inds = filter_inds[~remove]
        fg_bg_scores = fg_bg_scores[~remove]
        embed = embed[~remove]
        entropies = entropies[~remove]
        sigmoid_logit_scores = [sls[~remove] for sls in sigmoid_logit_scores]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh) # default: nms_thresh = 0.5
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    all_scores = all_scores[keep]
    fg_bg_scores = fg_bg_scores[keep]
    embed = embed[keep]
    entropies = entropies[keep]
    sigmoid_logit_scores = [sls[keep] for sls in sigmoid_logit_scores]

    assert all_scores.size(0) == scores.size(0)

    # 2.5. Convert some predictions into unknown category, after NMS
    if after_nms:
        scores, filter_inds = convert_known_into_unknown_entropy(scores, filter_inds, entropies, entropy_threshold, unk_label_ind)
        # Apply NMS to remove duplicated unknown
        keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh) # default: nms_thresh = 0.5
        if topk_per_image >= 0:
            keep = keep[:topk_per_image]
        boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
        all_scores = all_scores[keep]
        fg_bg_scores = fg_bg_scores[keep]
        embed = embed[keep]
        sigmoid_logit_scores = [sls[keep] for sls in sigmoid_logit_scores]
            
    # apply nms between known classes and unknown class for visualization.
    if vis_iou_thr < 1.0:
        boxes, scores, filter_inds = unknown_aware_nms(boxes, scores, filter_inds, ukn_class_id=unk_label_ind, iou_thr=vis_iou_thr)

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.all_scores = all_scores
    result.fg_bg_scores = fg_bg_scores
    result.pred_classes = filter_inds[:, 1]
    if sigmoid_logit_scores[0].mean() != -1: # if sigmoid_logit_scores was None, all elements should be -1.
        sig_scores, sig_labels = calc_sigmoid_label_scores(sigmoid_logit_scores)
        result.sigmoid_scores = sig_scores
        result.sigmoid_label_index = sig_labels
    if extracts_embed:
        result.embed = embed
    return result, filter_inds[:, 0]


def fast_rcnn_inference_single_image_entropy_summation(
    boxes,
    scores,
    embed,
    objectness,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    vis_iou_thr: float,
    entropy_threshold: float,
    after_nms: bool,
    compare_bg: bool = False,
    bg_to_unk: bool = False,
    sigmoid_logit_scores = None,
    extracts_embed = False
):
    if sigmoid_logit_scores is None:
        sigmoid_logit_scores = [-torch.ones_like(scores)]
        
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        sigmoid_logit_scores = [sls[valid_mask] for sls in sigmoid_logit_scores]
    
    fg_bg_scores = scores.clone() # [N, K+1] (not K+2; we don't expect that the OpenDet classifier uses this function)

    if bg_to_unk:
        # disable bg removal
        # scores = scores[:, :-1] 
        unk_label_ind = scores.shape[1]-1 # K-bg last index == NUM_CLASSES
        bg_label_ind = scores.shape[1]
    else:
        scores = scores[:, :-1] # remove bg class 
        unk_label_ind = scores.shape[1] # K-bg last index == NUM_CLASSES
    
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    
     # when comparing the confidence ratios, we consider background confidence, not only known classes's.
    if compare_bg:
        all_scores = fg_bg_scores[filter_inds[:, 0], :].clone() # [N, K+bg]
    else:  
        all_scores = scores[filter_inds[:, 0], :].clone() # [N, K]
    
    fg_bg_scores = fg_bg_scores[filter_inds[:, 0], :]
    embed = embed[filter_inds[:, 0], :]
    scores = scores[filter_mask]
    sigmoid_logit_scores = [sls[filter_inds[:, 0]] for sls in sigmoid_logit_scores]
    
    # transform bg_label_ind-1 into bg_label_ind 
    if bg_to_unk:
        bg_mask = filter_inds[:, 1] == bg_label_ind - 1
        filter_inds[bg_mask, 1] = bg_label_ind
  
    entropies = Categorical(probs=all_scores).entropy()
  
    # [sumup entropy of all bbs overlapping each other]
    overlap_entropies = sum_softmax_entropy(all_scores, boxes, 0.7)
    
    # 1.5. Convert some predictions into unknown category, before NMS
    if not after_nms:
        scores, filter_inds = convert_known_into_unknown_entropy(scores, filter_inds, overlap_entropies, entropy_threshold, unk_label_ind)
  
    # remove all BG predictions
    if bg_to_unk:
        remove = (filter_inds[:,1] == bg_label_ind)
        scores = scores[~remove]
        all_scores = all_scores[~remove]
        boxes = boxes[~remove]
        filter_inds = filter_inds[~remove]
        fg_bg_scores = fg_bg_scores[~remove]
        embed = embed[~remove]
        sigmoid_logit_scores = [sls[~remove] for sls in sigmoid_logit_scores]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh) # default: nms_thresh = 0.5
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    all_scores = all_scores[keep]
    fg_bg_scores = fg_bg_scores[keep]
    embed = embed[keep]
    sigmoid_logit_scores = [sls[keep] for sls in sigmoid_logit_scores]

    assert all_scores.size(0) == scores.size(0)

    # 2.5. Convert some predictions into unknown category, after NMS
    if after_nms:
        scores, filter_inds = convert_known_into_unknown_entropy(scores, filter_inds, overlap_entropies, entropy_threshold, unk_label_ind)
        # Apply NMS to remove duplicated unknown
        keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh) # default: nms_thresh = 0.5
        if topk_per_image >= 0:
            keep = keep[:topk_per_image]
        boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
        all_scores = all_scores[keep]
        fg_bg_scores = fg_bg_scores[keep]
        embed = embed[keep]
        sigmoid_logit_scores = [sls[keep] for sls in sigmoid_logit_scores]
        
    # apply nms between known classes and unknown class for visualization.
    if vis_iou_thr < 1.0:
        boxes, scores, filter_inds = unknown_aware_nms(boxes, scores, filter_inds, ukn_class_id=unk_label_ind, iou_thr=vis_iou_thr)

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.all_scores = all_scores
    result.fg_bg_scores = fg_bg_scores
    result.pred_classes = filter_inds[:, 1]
    if sigmoid_logit_scores[0].mean() != -1: # if sigmoid_logit_scores was None, all elements should be -1.
        sig_scores, sig_labels = calc_sigmoid_label_scores(sigmoid_logit_scores)
        result.sigmoid_scores = sig_scores
        result.sigmoid_label_index = sig_labels
    if extracts_embed:
        result.embed = embed
    return result, filter_inds[:, 0]



def calc_sigmoid_label_scores(sigmoid_logit):
    scores, labels = [], []
    for i in range(sigmoid_logit.__len__()):
        lvl_logit = sigmoid_logit[i]
        lvl_sig = lvl_logit.sigmoid()
        s, l = lvl_sig.topk(3, dim=1)
        scores.append(s)
        labels.append(l)
    scores = torch.cat(scores, dim=1)
    labels = torch.cat(labels, dim=1)
    return scores, labels


def convert_known_into_unknown_entropy(scores, filter_inds, entropies, entropy_thresh, unk_label_ind):
    assert scores.size(0) == filter_inds.size(0) == entropies.size(0)
    num_preds = scores.size(0)
    if num_preds == 0:
        return scores, filter_inds
        
    for b in range(num_preds):
        entropy = entropies[b]
        entropy_cond = entropy > entropy_thresh
        
        if entropy_cond:
            # update confidence into unknown score
            # unk_score = torch.min(torch.ones(1).type_as(scores), value.sum())
            unk_score = entropy
            scores[b] = unk_score
            # update cls label into unknown's
            filter_inds[b,1] = unk_label_ind
    return scores, filter_inds


def unknown_aware_nms(boxes, scores, labels, ukn_class_id=80, iou_thr=0.9):
    u_inds = labels[:, 1] == ukn_class_id
    k_inds = ~u_inds
    if k_inds.sum() == 0 or u_inds.sum() == 0:
        return boxes, scores, labels

    k_boxes, k_scores, k_labels = boxes[k_inds], scores[k_inds], labels[k_inds]
    u_boxes, u_scores, u_labels = boxes[u_inds], scores[u_inds], labels[u_inds]

    ious = pairwise_iou(Boxes(k_boxes), Boxes(u_boxes)) # xyxy
    mask = torch.ones((ious.size(0), ious.size(1), 2), device=ious.device)
    inds = (ious > iou_thr).nonzero()
    if not inds.numel():
        return boxes, scores, labels

    for [ind_x, ind_y] in inds:
        if k_scores[ind_x] >= u_scores[ind_y]:
            mask[ind_x, ind_y, 1] = 0 # k > u
        else:
            mask[ind_x, ind_y, 0] = 0 # k < u

    k_inds = mask[..., 0].mean(dim=1) == 1
    u_inds = mask[..., 1].mean(dim=0) == 1

    k_boxes, k_scores, k_labels = k_boxes[k_inds], k_scores[k_inds], k_labels[k_inds]
    u_boxes, u_scores, u_labels = u_boxes[u_inds], u_scores[u_inds], u_labels[u_inds]

    boxes = torch.cat([k_boxes, u_boxes])
    scores = torch.cat([k_scores, u_scores])
    labels = torch.cat([k_labels, u_labels])

    return boxes, scores, labels


# =================================================================================================


def fast_rcnn_inference_proposer(
    boxes: List[torch.Tensor],
    objectness: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    objectness_threshold: float, 
    nms_thresh: float, # 0.5
    topk_per_image: int, # 100
    vis_iou_thr: float = 1.0,
):
    """
    Based on fast_rcnn_inference, but without class predictions. 
    For NMS, instead of confidence scores, use objectness score (sigmoid). Thus, this process is class-agnostic NMS.
    """
    result_per_image = [
        fast_rcnn_inference_single_image_proposer(
            boxes_per_image, objectness_per_image, image_shape, objectness_threshold, nms_thresh, topk_per_image, vis_iou_thr,
        )
        for objectness_per_image, boxes_per_image, image_shape in zip(objectness, boxes, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image_proposer(
    boxes,
    objectness,
    image_shape: Tuple[int, int],
    obj_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    vis_iou_thr: float,
):
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(objectness).all(dim=0)

    if not valid_mask.all():
        boxes = boxes[valid_mask]
        objectness = objectness[valid_mask]
    
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = objectness > obj_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero() # [index of ROI, class_id]
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    objectness = objectness[filter_mask]
    
    pseudo_labels = torch.ones(filter_inds.__len__()).type_as(filter_inds)
    keep = batched_nms(boxes, objectness, pseudo_labels, nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, objectness, pseudo_labels = boxes[keep], objectness[keep], pseudo_labels[keep]

    # # apply nms between known classes and unknown class for visualization.
    # if vis_iou_thr < 1.0:
    #     boxes, scores, filter_inds = unknown_aware_nms(boxes, scores, filter_inds, ukn_class_id=unk_label_ind, iou_thr=vis_iou_thr)

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = objectness
    result.pred_classes = pseudo_labels
    return result, None