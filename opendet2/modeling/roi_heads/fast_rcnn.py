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
from detectron2.config import configurable
from detectron2.layers import (ShapeSpec, batched_nms, cat, cross_entropy,
                               nonzero_tuple)
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import (FastRCNNOutputLayers,
                                                     _log_classification_stats)
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.structures.boxes import matched_boxlist_iou
#  fast_rcnn_inference)
from detectron2.utils import comm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from ..layers import MLP
# from ..losses import ICLoss, UPLoss
from ..losses import ICLoss, UPLoss

import random

from .inference_utils import fast_rcnn_inference, fast_rcnn_inference_MSP, fast_rcnn_inference_ratio_ud, fast_rcnn_inference_entropy

ROI_BOX_OUTPUT_LAYERS_REGISTRY = Registry("ROI_BOX_OUTPUT_LAYERS")
ROI_BOX_OUTPUT_LAYERS_REGISTRY.__doc__ = """
ROI_BOX_OUTPUT_LAYERS
"""

logger = logging.getLogger(__name__)


def build_roi_box_output_layers(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_BOX_HEAD.OUTPUT_LAYERS
    return ROI_BOX_OUTPUT_LAYERS_REGISTRY.get(name)(cfg, input_shape)


@ROI_BOX_OUTPUT_LAYERS_REGISTRY.register()
class CosineFastRCNNOutputLayers(FastRCNNOutputLayers):
    @configurable
    def __init__(
        self,
        *args,
        scale: int = 20,
        vis_iou_thr: float = 1.0,
        **kargs,
    ):
        super().__init__(*args, **kargs)
        # prediction layer for num_classes foreground classes and one background class (hence + 1)
        self.cls_score = nn.Linear(
            self.cls_score.in_features, self.num_classes + 1, bias=False)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        # scaling factor
        self.scale = scale        
        self.vis_iou_thr = vis_iou_thr
        
    def prepare_hie_classifier(self, in_features, hie_clf_num_classes):
        num_hie = hie_clf_num_classes.__len__()
        per_hie_in_features = in_features//num_hie
        hie_in_features_list = [per_hie_in_features*i for i in range(num_hie, 0, -1)] # [3072, 2048, 1024]
        for i, hie_f in enumerate(hie_in_features_list):
            classifier = nn.Linear(
                hie_f, hie_clf_num_classes[i] + 1, bias=False
            )
            setattr(self, 'hie_cls_score_{}'.format(i+1), classifier)       
        return

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape) # **kargs
        ret['scale'] = cfg.MODEL.ROI_HEADS.COSINE_SCALE
        ret['vis_iou_thr'] = cfg.MODEL.ROI_HEADS.VIS_IOU_THRESH
        return ret


    def forward_cossim(self, cls_x, proposals=None):
        """
        OpenDet default:
        L2 normalize both input and classifier's weight, and get outputs as cosine similarity.
        """
        x_norm = torch.norm(cls_x, p=2, dim=1).unsqueeze(1).expand_as(cls_x)
        # check L2-norm of cls_x
        # print('norm: {} - {}'.format(x_norm.min(), x_norm.max()))
        x_normalized = cls_x.div(x_norm + 1e-5)

        # normalize weight
        temp_norm = (
            torch.norm(self.cls_score.weight.data, p=2, dim=1)
            .unsqueeze(1)
            .expand_as(self.cls_score.weight.data)
        )
        self.cls_score.weight.data = self.cls_score.weight.data.div(
            temp_norm + 1e-5
        )
        cos_dist = self.cls_score(x_normalized) # -1 ~ +1 ; cosine similarity
        
        # if any metric learning process (e.g., arcface, cosface, ...)
        scores = self.logit_scale.exp() * cos_dist # scaling

        return scores


    def forward(self, feats):
        # support shared & sepearte head
        if isinstance(feats, tuple):
            reg_x, cls_x = feats
        else:
            reg_x = cls_x = feats

        if reg_x.dim() > 2:
            reg_x = torch.flatten(reg_x, start_dim=1)
            cls_x = torch.flatten(cls_x, start_dim=1)

        x_norm = torch.norm(cls_x, p=2, dim=1).unsqueeze(1).expand_as(cls_x)
        x_normalized = cls_x.div(x_norm + 1e-5)

        # normalize weight
        temp_norm = (
            torch.norm(self.cls_score.weight.data, p=2, dim=1)
            .unsqueeze(1)
            .expand_as(self.cls_score.weight.data)
        )
        self.cls_score.weight.data = self.cls_score.weight.data.div(
            temp_norm + 1e-5
        )
        cos_dist = self.cls_score(x_normalized)
        scores = self.scale * cos_dist
        proposal_deltas = self.bbox_pred(reg_x)

        return scores, proposal_deltas
    
    
    def cat_detach(self, cls_x_list):
        # detach except for the first element / [x1, x2.detach(), x3.detach(), ..., xn.detach()]
        if cls_x_list.__len__() == 0:
            return cls_x_list[0] # without detach
        return torch.cat([x if i==0 else x.detach() for i,x in enumerate(cls_x_list)], dim=1)

    def predict_boxes(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        if not len(proposals):
            return []
        proposal_deltas = predictions[1]
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat(
            [p.proposal_boxes.tensor for p in proposals], dim=0)
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)


    def predict_probs(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        # default
        scores = predictions[0]

        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        
        embed = probs.clone()
        
        probs = probs.split(num_inst_per_image, dim=0)
        embed = embed.split(num_inst_per_image, dim=0)
        return probs, embed


    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        boxes = self.predict_boxes(predictions, proposals)
        scores, embed = self.predict_probs(predictions, proposals)
        
        # get objectness scores to decay
        num_inst_per_image = [len(p) for p in proposals]
        objectness = F.sigmoid(cat([p.objectness_logits for p in proposals]))
        objectness = objectness.split(num_inst_per_image, dim=0)
        
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            embed,
            objectness,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
            self.vis_iou_thr,
        )
        

    def inference_with_K_classifier(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Sub function of 'inference':
        Conduct unknown detection without additional (e.g., K+1) placeholder in a classifier.
        This should be called in some methods using K classifier : OE, MINUS, ...
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores, embed = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        
        # get objectness scores to decay
        num_inst_per_image = [len(p) for p in proposals]
        objectness = F.sigmoid(cat([p.objectness_logits for p in proposals]))
        objectness = objectness.split(num_inst_per_image, dim=0)
        
        # =========================================================
        if self.msp_threshold > 0:
            # use max softmax probability to detect unknown
            assert self.entropy_threshold < 0. and self.ratio_ud_unk_ratio < 0.
            return fast_rcnn_inference_MSP(
                boxes,
                scores,
                embed,
                objectness,
                image_shapes,
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_topk_per_image,
                self.vis_iou_thr,
                self.msp_threshold,
                self.after_nms,
                extracts_embed = self.extracts_embed,
            )
        elif self.ratio_ud_unk_ratio > 0:
            # unknown detection via top1/2 ratio
            assert self.entropy_threshold < 0. and self.msp_threshold < 0.
            return fast_rcnn_inference_ratio_ud(
                boxes,
                scores,
                embed,
                objectness,
                image_shapes,
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_topk_per_image,
                self.vis_iou_thr,
                self.ratio_ud_unk_ratio,
                self.after_nms,
                self.ratio_ud_compare_bg,
                self.ratio_ud_bg_to_unk,
                self.inference_wo_bg,
                extracts_embed = self.extracts_embed,
            )
        elif self.entropy_threshold > 0:
            # unknown detection via entropy
            assert self.ratio_ud_unk_ratio < 0. and self.msp_threshold < 0.
            return fast_rcnn_inference_entropy(
                boxes,
                scores,
                embed,
                objectness,
                image_shapes,
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_topk_per_image,
                self.vis_iou_thr,
                self.entropy_threshold,
                self.after_nms,
                self.ratio_ud_compare_bg,
                self.ratio_ud_bg_to_unk,
                self.inference_wo_bg,
                extracts_embed = self.extracts_embed,
            )
        else: # default | without unknown detection process
            return fast_rcnn_inference(
                boxes,
                scores,
                embed,
                objectness,
                image_shapes,
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_topk_per_image,
                self.vis_iou_thr,
                extracts_embed = self.extracts_embed,
            )



# === box predictor functions === 
@ROI_BOX_OUTPUT_LAYERS_REGISTRY.register()
class OpenDetFastRCNNOutputLayers(CosineFastRCNNOutputLayers):
    @configurable
    def __init__(
        self,
        *args,
        num_known_classes,
        max_iters,
        up_loss_start_iter,
        up_loss_sampling_metric,
        up_loss_topk,
        up_loss_alpha,
        up_loss_weight,
        ic_loss_out_dim,
        ic_loss_queue_size,
        ic_loss_in_queue_size,
        ic_loss_batch_iou_thr,
        ic_loss_queue_iou_thr,
        ic_loss_queue_tau,
        ic_loss_weight,
        **kargs
    ):
        super().__init__(*args, **kargs)
        self.num_known_classes = num_known_classes
        self.max_iters = max_iters

        self.up_loss = UPLoss(
            self.num_classes,
            sampling_metric=up_loss_sampling_metric,
            topk=up_loss_topk,
            alpha=up_loss_alpha
        )
        self.up_loss_start_iter = up_loss_start_iter
        self.loss_weight['loss_cls_up'] = up_loss_weight

        self.encoder = MLP(self.cls_score.in_features, ic_loss_out_dim)
        self.ic_loss_loss = ICLoss(tau=ic_loss_queue_tau)
        self.ic_loss_out_dim = ic_loss_out_dim
        self.ic_loss_queue_size = ic_loss_queue_size
        self.ic_loss_in_queue_size = ic_loss_in_queue_size
        self.ic_loss_batch_iou_thr = ic_loss_batch_iou_thr
        self.ic_loss_queue_iou_thr = ic_loss_queue_iou_thr
        self.loss_weight['loss_cls_ic'] = ic_loss_weight

        self.register_buffer('queue', torch.zeros(
            self.num_known_classes, ic_loss_queue_size, ic_loss_out_dim)) # [K, Q, D]
        self.register_buffer('queue_label', torch.empty(
            self.num_known_classes, ic_loss_queue_size).fill_(-1).long()) # [K, Q]
        self.register_buffer('queue_ptr', torch.zeros(
            self.num_known_classes, dtype=torch.long)) # [K]
                
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update({
            'num_known_classes': cfg.MODEL.ROI_HEADS.NUM_KNOWN_CLASSES,
            "max_iters": cfg.SOLVER.MAX_ITER,

            "up_loss_start_iter": cfg.UPLOSS.START_ITER,
            "up_loss_sampling_metric": cfg.UPLOSS.SAMPLING_METRIC,
            "up_loss_topk": cfg.UPLOSS.TOPK,
            "up_loss_alpha": cfg.UPLOSS.ALPHA,
            "up_loss_weight": cfg.UPLOSS.WEIGHT,

            "ic_loss_out_dim": cfg.ICLOSS.OUT_DIM,
            "ic_loss_queue_size": cfg.ICLOSS.QUEUE_SIZE,
            "ic_loss_in_queue_size": cfg.ICLOSS.IN_QUEUE_SIZE,
            "ic_loss_batch_iou_thr": cfg.ICLOSS.BATCH_IOU_THRESH,
            "ic_loss_queue_iou_thr": cfg.ICLOSS.QUEUE_IOU_THRESH,
            "ic_loss_queue_tau": cfg.ICLOSS.TEMPERATURE,
            "ic_loss_weight": cfg.ICLOSS.WEIGHT,
        })
        return ret


    def get_up_loss(self, scores, gt_classes):
        # start up loss after several warmup iters
        storage = get_event_storage()
        if storage.iter > self.up_loss_start_iter:
            if scores.__len__() == 0:
                loss_cls_up = scores.new_tensor(0.0)
            else: # calc loss
                loss_cls_up = self.up_loss(scores, gt_classes)
        else:
            loss_cls_up = scores.new_tensor(0.0)
        # return {"loss_cls_up": self.up_loss_weight * loss_cls_up}
        return {"loss_cls_up": loss_cls_up}


    def get_ic_loss(self, feat, gt_classes, ious):
        # select foreground and iou > thr instance in a mini-batch
        pos_inds = (ious > self.ic_loss_batch_iou_thr) & (gt_classes != self.num_classes)
        
        feat, gt_classes = feat[pos_inds], gt_classes[pos_inds]

        queue = self.queue.reshape(-1, self.ic_loss_out_dim)
        queue_label = self.queue_label.reshape(-1)
        queue_inds = queue_label != -1  # filter empty queue
        queue, queue_label = queue[queue_inds], queue_label[queue_inds]

        if feat.__len__() == 0:
            loss_ic_loss = feat.new_tensor(0.0)
        else: # calc loss
            loss_ic_loss = self.ic_loss_loss(feat, gt_classes, queue, queue_label)
        # loss_ic_loss = self.ic_loss_loss(feat, gt_classes, queue, queue_label)
        # loss decay
        storage = get_event_storage()
        decay_weight = 1.0 - storage.iter / self.max_iters # small weight on large iter
        # return {"loss_cls_ic": self.ic_loss_weight * decay_weight * loss_ic_loss}
        return {"loss_cls_ic": decay_weight * loss_ic_loss}
    

    @torch.no_grad()
    def _dequeue_and_enqueue(self, feat, gt_classes, ious, iou_thr=0.7):
        # 1. gather variable
        # print('start {}'.format(comm.get_local_rank()))
        feat = self.concat_all_gather(feat)
        gt_classes = self.concat_all_gather(gt_classes)
        ious = self.concat_all_gather(ious)
        # print('pending in rank {}...'.format(comm.get_local_rank()))
        # print()
        # 2. filter by iou and obj, remove bg
        keep = (ious > iou_thr) & (gt_classes != self.num_classes)
        feat, gt_classes = feat[keep], gt_classes[keep]
        for i in range(self.num_known_classes):
            ptr = int(self.queue_ptr[i]) # current num of queues on the class=i
            cls_ind = gt_classes == i # index of proposal features whose class=i 
            cls_feat, cls_gt_classes = feat[cls_ind], gt_classes[cls_ind] # filter 
            # 3. sort by similarity, low sim ranks first
            cls_queue = self.queue[i, self.queue_label[i] != -1] # In queue, which index is valid (in queued)
            _, sim_inds = F.cosine_similarity(
                cls_feat[:, None], cls_queue[None, :], dim=-1).mean(dim=1).sort()
            top_sim_inds = sim_inds[:self.ic_loss_in_queue_size] # add in queue from low-similarity features 
            cls_feat, cls_gt_classes = cls_feat[top_sim_inds], cls_gt_classes[top_sim_inds]
            # 4. in queue (old+new < 256, add all/ ) 
            batch_size = cls_feat.size(0) if ptr + cls_feat.size(0) <= self.ic_loss_queue_size \
                                          else self.ic_loss_queue_size - ptr
            self.queue[i, ptr:ptr+batch_size] = cls_feat[:batch_size]
            self.queue_label[i, ptr:ptr + batch_size] = cls_gt_classes[:batch_size]

            ptr = ptr + batch_size if ptr + batch_size < self.ic_loss_queue_size else 0
            self.queue_ptr[i] = ptr
                   

    @torch.no_grad()
    def concat_all_gather(self, tensor):
        world_size = comm.get_world_size()
        # single GPU, directly return the tensor
        if world_size == 1:
            return tensor
        # multiple GPUs, gather tensors
        tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
        return output


    def forward(self, feats, proposals=None, apply_mfmixup=False):
        # support shared & sepearte head
        if isinstance(feats, tuple):
            reg_x, cls_x = feats
        else:
            reg_x = cls_x = feats

        if reg_x.dim() > 2:
            reg_x = torch.flatten(reg_x, start_dim=1)
            cls_x = torch.flatten(cls_x, start_dim=1)

        # encode feature with MLP
        mlp_feat = self.encoder(cls_x)
        scores = self.forward_cossim(cls_x, proposals) # calculate cosine similarity W^T x
        proposal_deltas = self.bbox_pred(reg_x)
        
        return scores, proposal_deltas, mlp_feat
    

    def losses(self, predictions, proposals, input_features=None):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas, mlp_feat = predictions
        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat(
                [p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty(
                (0, 4), device=proposal_deltas.device)

        losses = {
            "loss_cls_ce": cross_entropy(scores, gt_classes, reduction="mean"),
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }

        # up loss (Unknown Probability Learner)
        losses.update(self.get_up_loss(scores, gt_classes))

        ious = cat([p.iou for p in proposals], dim=0)
        # we first store feats in the queue, then compute loss
        self._dequeue_and_enqueue(mlp_feat, gt_classes, ious, iou_thr=self.ic_loss_queue_iou_thr)        
        losses.update(self.get_ic_loss(mlp_feat, gt_classes, ious))

        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
