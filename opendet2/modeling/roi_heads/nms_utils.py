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

def get_ious(box1, box2):
    if box1.ndim == 1:
        box1.unsqueeze_(0)
    if box2.ndim == 1:
        box2.unsqueeze_(0)
    
    _iscrowd = [0 for i in range(len(box2))]
    return maskUtils.iou(box1.data.cpu().numpy(), box2.data.cpu().numpy(), _iscrowd)


def sum_softmax_entropy(score_dists, boxes, iou_th):
    # entropies = Categorical(probs=score_dists).entropy()
    if len(boxes) <= 1:
        return Categorical(probs=score_dists).entropy().flatten()
    
    res = []
    for i,b in enumerate(boxes):
        score_self = score_dists[i][None, :]
        other_scores = torch.cat([score_dists[:i], score_dists[i+1:]], dim=0)
        other_boxes = torch.cat([boxes[:i], boxes[i+1:]], dim=0)
        ious = get_ious(b, other_boxes)
        
        overlaps = ious > iou_th
        # overlap_ious = ious[overlaps]
        
        overlap_scores = score_self + other_scores[overlaps].sum(dim=0, keepdims=True)
        # overlap_entropies = entropies[overlaps] # 1.
        overlap_entropies = Categorical(probs=overlap_scores).entropy() # 2.
        res.append(overlap_entropies)
    return torch.stack(res).flatten()
