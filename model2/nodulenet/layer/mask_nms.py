from . import *
import torch
import numpy as np
# import torch.nn.functional as F



def mask_nms(cfg, mode, mask_logits, crop_boxes, inputs):
    nms_overlap_threshold   = cfg['mask_test_nms_overlap_threshold']
    batch_size, _, depth, height, width = inputs.size() #original image width
    num_class = cfg['num_class']
    keep_ids = []

    for b in range(batch_size):
        crop_boxes_batch = crop_boxes[crop_boxes[:, 0] == b]
        crop_boxes_batch = crop_boxes_batch[:, 1:]
        n = len(crop_boxes_batch)
        cur = 0
        visited = [False for _ in range(n)]
        while cur < n:
            if visited[cur]:
                cur += 1
                continue

            visited[cur] = True
            keep_ids.append(cur)
            mask1 = mask_logits2probs(mask_logits[cur])
            for i in range(cur + 1, n):
                mask2 = mask_logits2probs(mask_logits[i])
                if mask_iou(mask1, mask2) > nms_overlap_threshold:
                    visited[i] = True
            
            cur += 1

    return keep_ids


def mask_iou(mask1, mask2):
    return float(np.logical_and(mask1, mask2).sum()) / np.logical_or(mask1, mask2).sum()


def mask_logits2probs(mask):
    mask = (torch.sigmoid(mask) > 0.5).cpu().numpy().astype(np.uint8)
    return mask


# def mask_nms(cfg, mode, mask_logits, crop_boxes, inputs):
#     nms_overlap_threshold   = cfg['mask_test_nms_overlap_threshold']
#     batch_size, _, depth, height, width = inputs.size() #original image width
#     num_class = cfg['num_class']
#     keep_ids = []

#     for b in range(batch_size):
#         crop_boxes_batch = crop_boxes[crop_boxes[:, 0] == b]
#         crop_boxes_batch = crop_boxes_batch[:, 1:]
#         n = len(crop_boxes_batch)
#         cur = 0
#         visited = [False for _ in range(n)]
#         while cur < n:
#             if visited[cur]:
#                 cur += 1
#                 continue

#             visited[cur] = True
#             keep_ids.append(cur)
#             mask1 = mask_logits[cur]
#             # mask1 = mask_logits2probs(mask_logits[cur])
#             for i in range(cur + 1, n):
#                 # mask2 = mask_logits2probs(mask_logits[i])
#                 mask2 = mask_logits[i]
#                 if mask_iou(mask1, mask2) > nms_overlap_threshold:
#                     visited[i] = True
            
#             cur += 1
#     return keep_ids


# def mask_iou(mask1, mask2):
#     z_compare = torch.reshape(torch.stack(torch.meshgrid([mask1[0], mask2[0]]), dim=0), (2, -1))
#     z_compare = torch.where(z_compare[0]==z_compare[1], 1, 0)
#     y_compare = torch.reshape(torch.stack(torch.meshgrid([mask1[1], mask2[1]]), dim=0), (2, -1))
#     y_compare = torch.where(y_compare[0]==y_compare[1], 1, 0)
#     comp1 = torch.logical_and(z_compare, y_compare)

#     # TODO: below z_compare should be x_compare, change it temporally because CUDA out of memory Error
#     z_compare = torch.reshape(torch.stack(torch.meshgrid([mask1[2], mask2[2]]), dim=0), (2, -1))
#     z_compare = torch.where(z_compare[0]==z_compare[1], 1, 0)
#     position_compare = torch.logical_and(
#         comp1, torch.logical_and(z_compare, y_compare))

#     num1 = mask1[0].shape[0]
#     num2 = mask2[0].shape[0]
#     intersection  = torch.sum(position_compare).float()
#     union = num1 + num2 - intersection
#     union = union.float()
#     iou = intersection / union
#     return iou


# def mask_logits2probs(mask):
#     mask = (torch.sigmoid(mask) > 0.5).cpu().numpy().astype(np.uint8)
#     return mask