import copy

import torch

from model2.nodulenet.layer import *
from model2.nodulenet.config import net_config as config
from model2.nodulenet.utils.util import center_box_to_coord_box, ext2factor, clip_boxes, crop_boxes2mask_single
from model2.nodulenet.nodule_net import crop_mask_regions, rcnn_crop


def model_inference(cfg, inputs, feature_net, rpn, rcnn_head, mask_head):
    mode = 'eval'
    use_rcnn = True
    use_mask = True

    # Image feature
    features, feat_4 = feature_net(inputs); #print('fs[-1] ', fs[-1].shape)
    fs = features[-1]

    # RPN proposals
    rpn_logits_flat, rpn_deltas_flat = rpn(fs)

    b,D,H,W,_,num_class = rpn_logits_flat.shape

    rpn_logits_flat = rpn_logits_flat.view(b, -1, 1);#print('rpn_logit ', rpn_logits_flat.shape)
    rpn_deltas_flat = rpn_deltas_flat.view(b, -1, 6);#print('rpn_delta ', rpn_deltas_flat.shape)

    rpn_window  = make_rpn_windows(fs, cfg)
    rpn_proposals = []
    if use_rcnn:
        rpn_proposals = rpn_nms(cfg, mode, inputs, rpn_window,
                rpn_logits_flat, rpn_deltas_flat)
        # print 'length of rpn proposals', rpn_proposals.shape

    # RCNN proposals
    detections = copy.deepcopy(rpn_proposals)
    mask_probs, mask_targets = [], []
    crop_boxes = []
    if use_rcnn:
        if len(rpn_proposals) > 0:
            rcnn_crops = rcnn_crop(feat_4, inputs, rpn_proposals)
            rcnn_logits, rcnn_deltas = rcnn_head(rcnn_crops)
            detections, keeps = rcnn_nms(
                cfg, mode, inputs, rpn_proposals, 
                rcnn_logits, rcnn_deltas
            ) 

        # pred_mask = np.zeros(list(inputs.shape[2:]))
        if use_mask and len(detections):
            # keep batch index, z, y, x, d, h, w, class
            if len(detections):
                crop_boxes = detections[:, [0, 2, 3, 4, 5, 6, 7, 8]].cpu().numpy().copy()
                crop_boxes[:, 1:-1] = center_box_to_coord_box(crop_boxes[:, 1:-1])
                crop_boxes = crop_boxes.astype(np.int32)
                # crop_boxes[:, 4:-1] = crop_boxes[:, 4:-1] + np.ones_like(crop_boxes[:, 4:-1])
                crop_boxes[:, 1:-1] = ext2factor(crop_boxes[:, 1:-1], 4)
                crop_boxes[:, 1:-1] = clip_boxes(crop_boxes[:, 1:-1], inputs.shape[2:])
            
            # Make sure to keep feature maps not splitted by data parallel
            features = [t.unsqueeze(0).expand(torch.cuda.device_count(), -1, -1, -1, -1, -1) for t in features]
            mask_probs = mask_head(torch.from_numpy(crop_boxes).cuda(), features)

            mask_keep = mask_nms(cfg, mode, mask_probs, crop_boxes, inputs)
            crop_boxes = crop_boxes[mask_keep]
            detections = detections[mask_keep]
            # mask_probs = mask_probs[mask_keep]
            out_masks = []
            for keep_idx in mask_keep:
                out_masks.append(mask_probs[keep_idx])
            mask_probs = out_masks
            
            pred_mask = crop_mask_regions(mask_probs, crop_boxes, features[0][0].shape)

            # # segments = [torch.sigmoid(m).cpu().numpy() > 0.5 for m in mask_probs]
            # segments = [torch.sigmoid(m) > 0.5 for m in mask_probs]
            # pred_mask = crop_boxes2mask_single(crop_boxes[:, 1:], segments, inputs.shape[2:])
    # TODO: correctly get the num_class and batch size dimension
    # pred_mask = pred_mask[None, None]
    return pred_mask