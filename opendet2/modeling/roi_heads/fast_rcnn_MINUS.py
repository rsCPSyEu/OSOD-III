# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict, List, Tuple, Union
import logging
import numpy as np
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import configurable
from detectron2.layers import cat, cross_entropy
from detectron2.modeling.roi_heads.fast_rcnn import (FastRCNNOutputLayers,
                                                     _log_classification_stats)
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils import comm
from detectron2.utils.events import get_event_storage
from torch import nn
from torch.nn import functional as F

from ..layers import MLP
from ..losses import ICLoss, UPLoss

from itertools import combinations

from .fast_rcnn import CosineFastRCNNOutputLayers
from .fast_rcnn import ROI_BOX_OUTPUT_LAYERS_REGISTRY

logger = logging.getLogger(__name__)

@ROI_BOX_OUTPUT_LAYERS_REGISTRY.register()
class OpenDetFastRCNNOutputLayers_MINUS(CosineFastRCNNOutputLayers):
    '''
    From OpenDetFastRCNNOutputLayers: 
        Replace K+1 classifier with K classifier
        Conduct unknown detection usin ratio
    '''
    @configurable
    def __init__(
        self,
        *args,
        num_known_classes,
        max_iters,
        ic_loss_out_dim,
        ic_loss_queue_size,
        ic_loss_in_queue_size,
        ic_loss_batch_iou_thr,
        ic_loss_queue_iou_thr,
        ic_loss_queue_tau,
        ic_loss_weight,
        # for MSP
        msp_threshold,
        # ratio unknown detection (without K+1 classifier)
        ratio_ud_unk_ratio,
        ratio_ud_temp,
        after_nms,
        ratio_ud_compare_bg,
        ratio_ud_bg_to_unk,
        entropy_threshold,
        **kargs
    ):
        super().__init__(*args, **kargs)
        self.num_known_classes = num_known_classes
        self.max_iters = max_iters

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
            
        # Args to switch unknown detection using ratio (without K+1 classifier)
        self.ratio_ud_unk_ratio = ratio_ud_unk_ratio
        self.ratio_ud_temp = ratio_ud_temp
        self.after_nms = after_nms
        self.ratio_ud_compare_bg = ratio_ud_compare_bg
        self.ratio_ud_bg_to_unk = ratio_ud_bg_to_unk
            
        # maximum softmax probability threshold
        self.msp_threshold = msp_threshold
                    
        # entropy threshold
        self.entropy_threshold = entropy_threshold
        

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update({
            'num_known_classes': cfg.MODEL.ROI_HEADS.NUM_KNOWN_CLASSES,
            "max_iters": cfg.SOLVER.MAX_ITER,
            
            "ic_loss_out_dim": cfg.ICLOSS.OUT_DIM,
            "ic_loss_queue_size": cfg.ICLOSS.QUEUE_SIZE,
            "ic_loss_in_queue_size": cfg.ICLOSS.IN_QUEUE_SIZE,
            "ic_loss_batch_iou_thr": cfg.ICLOSS.BATCH_IOU_THRESH,
            "ic_loss_queue_iou_thr": cfg.ICLOSS.QUEUE_IOU_THRESH,
            "ic_loss_queue_tau": cfg.ICLOSS.TEMPERATURE,
            "ic_loss_weight": cfg.ICLOSS.WEIGHT,
            
            "msp_threshold": cfg.MSP.TH,
            
            "ratio_ud_unk_ratio": cfg.RATIO_UD.UNK_RATIO,
            "ratio_ud_temp": cfg.RATIO_UD.TEMP,
            "after_nms": cfg.RATIO_UD.AFTER_NMS,
            "ratio_ud_compare_bg": cfg.RATIO_UD.COMPARE_BG,
            "ratio_ud_bg_to_unk": cfg.RATIO_UD.BG_TO_UNK,
            
            "entropy_threshold": cfg.ENTROPY_TH,
        })
        return ret
    

    def get_ic_loss(self, feat, gt_classes, ious):
        # select foreground and iou > thr instance in a mini-batch
        pos_inds = (ious > self.ic_loss_batch_iou_thr) & (gt_classes != self.num_classes)
        
        feat, gt_classes = feat[pos_inds], gt_classes[pos_inds]

        queue = self.queue.reshape(-1, self.ic_loss_out_dim)
        queue_label = self.queue_label.reshape(-1)
        queue_inds = queue_label != -1  # filter empty queue
        queue, queue_label = queue[queue_inds], queue_label[queue_inds]

        loss_ic_loss = self.ic_loss_loss(feat, gt_classes, queue, queue_label)
        # loss decay
        storage = get_event_storage()
        decay_weight = 1.0 - storage.iter / self.max_iters # small weight on large iter
        return {"loss_cls_ic": decay_weight * loss_ic_loss}


    @torch.no_grad()
    def _dequeue_and_enqueue(self, feat, gt_classes, ious, iou_thr=0.7):
        # 1. gather variable
        feat = self.concat_all_gather(feat)
        gt_classes = self.concat_all_gather(gt_classes)
        ious = self.concat_all_gather(ious)
        # 2. filter by iou and obj, remove bg
        keep = (ious > iou_thr) & (gt_classes != self.num_classes) # remove bg category's proposal
        feat, gt_classes = feat[keep], gt_classes[keep]
        # what does ptr mean?
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
        scores = self.forward_cossim(cls_x, proposals)
        proposal_deltas = self.bbox_pred(reg_x)
        
        # temperature
        if self.ratio_ud_temp > 0.:
            scores /= self.ratio_ud_temp # exp(x/T)
        
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
        scores, proposal_deltas, _ = predictions
                
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
            "loss_box_reg": self.box_reg_loss(proposal_boxes, gt_boxes, proposal_deltas, gt_classes),
        }

        # # we first store feats in the queue, then compute loss
        # if self.loss_weight.get('loss_cls_ic', 1.0) > 0.:
        #     self._dequeue_and_enqueue(mlp_feat, gt_classes, ious, iou_thr=self.ic_loss_queue_iou_thr)        
        #     losses.update(self.get_ic_loss(mlp_feat, gt_classes, ious))

        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
    

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        return self.inference_with_K_classifier(predictions, proposals)