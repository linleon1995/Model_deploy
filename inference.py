import copy
import sys
import os
import argparse

import numpy as np
import torch
import onnxruntime
import cc3d

from model2.nodulenet.layer.rpn_nms import make_rpn_windows, rpn_nms
from model2.nodulenet.layer.rcnn_nms import rcnn_nms
from model2.nodulenet.layer.mask_nms import mask_nms
from model2.nodulenet.config import config
from model2.nodulenet.utils.util import (
    center_box_to_coord_box, ext2factor, clip_boxes, crop_boxes2mask_single, pad2factor)
from model2.nodulenet.nodule_net import crop_mask_regions, NoduleNet
from model2.nodulenet.utils.LIDC.preprocess_TMH import (
    load_itk_image, preprocess_op_new, resample2)
from utils import timer_func, ONNX_inference_from_session
from nodule_to_nrrd import seg_nrrd_write


parser = argparse.ArgumentParser() 
parser.add_argument(
    '--input', type=str, required=True,  help='Input file')

parser.add_argument(
    '--model-dir', type=str, default=None, required=False, help='The ONNX model directory')

parser.add_argument(
    '--pred-dir', type=str, default=None, required=False, help='The nodule egmentation saving directory')


@timer_func
def model_inference(
    cfg, inputs, feature_net_session, rpn_session, rcnn_session, mask_session, 
    rcnn_crop, filename):
    mode = 'eval'
    use_rcnn = True
    use_mask = True
    max_bboxs = None

    # Image feature
    features = ONNX_inference_from_session(inputs, feature_net_session)

    fs = features[-2]
    feat_4 = features[-1]

    # RPN proposals
    rpn_logits, rpn_deltas = ONNX_inference_from_session(fs, rpn_session)

    b,D,H,W,_,num_class = rpn_logits.shape

    rpn_logits_flat = np.reshape(rpn_logits, (b, -1, 1))
    rpn_deltas_flat = np.reshape(rpn_deltas, (b, -1, 6))

    # TODO: consider just use Numpy
    fs = torch.from_numpy(fs)
    inputs = torch.from_numpy(inputs)
    rpn_logits_flat = torch.from_numpy(rpn_logits_flat)
    rpn_deltas_flat = torch.from_numpy(rpn_deltas_flat)
    feat_4 = torch.from_numpy(feat_4)

    rpn_window  = make_rpn_windows(fs, cfg)
    rpn_proposals = []
    if use_rcnn:
        rpn_proposals = rpn_nms(cfg, mode, inputs, rpn_window,
                rpn_logits_flat, rpn_deltas_flat)

    # RCNN proposals
    detections = copy.deepcopy(rpn_proposals)
    mask_probs, mask_targets = [], []
    crop_boxes = []
    if use_rcnn:
        if len(rpn_proposals) > 0:
            rcnn_crops = rcnn_crop(feat_4, inputs, rpn_proposals)
            # rcnn_logits, rcnn_deltas = rcnn_head(rcnn_crops)
            rcnn_crops = rcnn_crops.cpu().numpy()
            rcnn_logits, rcnn_deltas = ONNX_inference_from_session(rcnn_crops, rcnn_session)

            rcnn_logits = torch.from_numpy(rcnn_logits)
            rcnn_deltas = torch.from_numpy(rcnn_deltas)


            detections, keeps = rcnn_nms(
                cfg, mode, inputs, rpn_proposals, 
                rcnn_logits, rcnn_deltas, max_bboxs
            ) 

        pred_mask = np.zeros(list(inputs.shape[2:]))
        if use_mask and len(detections):
            # keep batch index, z, y, x, d, h, w, class
            if len(detections):
                crop_boxes = detections[:, [0, 2, 3, 4, 5, 6, 7, 8]].cpu().numpy().copy()
                crop_boxes[:, 1:-1] = center_box_to_coord_box(crop_boxes[:, 1:-1])
                crop_boxes = crop_boxes.astype(np.int32)
                crop_boxes[:, 1:-1] = ext2factor(crop_boxes[:, 1:-1], 4)
                crop_boxes[:, 1:-1] = clip_boxes(crop_boxes[:, 1:-1], inputs.shape[2:])
            
            mask_probs = []
            mask_probs_for_nms = []
            for detection in crop_boxes:
                b, z_start, y_start, x_start, z_end, y_end, x_end, cat = detection
                im = features[0][:, :, z_start:z_end, y_start:y_end, x_start:x_end]
                crop_f2 = features[1][:, :, z_start//2:z_end//2, y_start//2:y_end//2, x_start//2:x_end//2]
                crop_f4 = features[2][:, :, z_start//4:z_end//4, y_start//4:y_end//4, x_start//4:x_end//4]
                ort_inputs = {
                    mask_session.get_inputs()[0].name: crop_f4,
                    mask_session.get_inputs()[1].name: crop_f2,
                    mask_session.get_inputs()[2].name: im,
                    # mask_session.get_inputs()[3].name: features[3],
                }
                mask_prob = torch.from_numpy(mask_session.run(None, ort_inputs)[0][0, 0])
                # m = torch.zeros(features[0][0, 0].shape)
                # m[z_start:z_end, y_start:y_end, x_start:x_end] = mask_prob

                # TODO: cuda
                # TODO: optimize code
                mask_probs_for_nms.append((mask_prob, features[0][0, 0].shape))
                # mask_probs.append(mask_prob)

            mask_keep, mask_probs = mask_nms(cfg, mode, mask_probs_for_nms, crop_boxes, inputs)
            crop_boxes = crop_boxes[mask_keep]
            detections = detections[mask_keep]
            # mask_probs = mask_probs[mask_keep]

            out_masks = []
            weight = []
            for mask_prob in mask_probs:
                out_masks.append(mask_prob.float())
                weight.append(mask_prob>0)
            pred_mask = torch.stack(out_masks, dim=0)
            weight = torch.stack(weight, dim=0)
            pred_mask = torch.sum(pred_mask, dim=0) / torch.sum(weight, dim=0)
            pred_mask = torch.where(torch.sigmoid(pred_mask)>0.5, 1, 0)
            
            # # pred_mask = crop_mask_regions(mask_probs, crop_boxes, features[0].shape)
            # mask_probs = crop_mask_regions(mask_probs, crop_boxes)
            # # segments = [torch.sigmoid(m) for m in mask_probs]
            # segments = [torch.sigmoid(m) > 0.5 for m in mask_probs]
            # pred_mask = crop_boxes2mask_single(crop_boxes[:, 1:], segments, inputs.shape[2:])
    
    pred_mask = pred_mask.numpy()
    pred_mask = cc3d.connected_components(pred_mask , 26)
    return pred_mask


def get_nodule_center(pred_volume):
    total_nodule_center = {}
    for nodule_id in np.unique(pred_volume)[1:]:
        zs, ys, xs = np.where(pred_volume==nodule_id)
        center_index, center_row, center_column = np.mean(zs), np.mean(ys), np.mean(xs)
        total_nodule_center[nodule_id] = {
            'Center': {'index': np.mean(center_index).astype('int32'), 
            'row': np.mean(center_row).astype('int32'), 
            'column': np.mean(center_column).astype('int32')}}
    return total_nodule_center
    

def crop_volume(volume, crop_range, crop_center):
    def get_interval(crop_range_dim, center, size_dim):
        begin = center - crop_range_dim//2
        end = center + crop_range_dim//2
        if begin < 0:
            begin, end = 0, end-begin
        elif end > size_dim:
            modify_distance = end - size_dim + 1
            begin, end = begin-modify_distance, size_dim-1
        # print(crop_range_dim, center, size_dim, begin, end)
        assert end-begin == crop_range_dim, \
            f'Actual cropping range {end-begin} not fit the required cropping range {crop_range_dim}'
        return (begin, end)

    index_interval = get_interval(crop_range['index'], crop_center['index'], volume.shape[0])
    row_interval = get_interval(crop_range['row'], crop_center['row'], volume.shape[1])
    column_interval = get_interval(crop_range['column'], crop_center['column'], volume.shape[2])

    return volume[index_interval[0]:index_interval[1], 
                  row_interval[0]:row_interval[1], 
                  column_interval[0]:column_interval[1]]


@timer_func
def nodule_cls(raw_volume, pred_volume_category, onnx_session):
    """AI is creating summary for nodule_cls

    Args:
        raw_volume ([D, H, W]): [description]
        pred_volume_category ([D, H, W]): [description]
        onnx_session: [description]

    Returns:
        [type]: [description]
    """
    pred_nodules = get_nodule_center(pred_volume_category)
    remove_nodule_ids = []
    crop_range = {'index': 32, 'row': 64, 'column': 64}
    for nodule_id in list(pred_nodules):
        crop_raw_volume = crop_volume(raw_volume, crop_range, pred_nodules[nodule_id]['Center'])
        crop_raw_volume = np.expand_dims(crop_raw_volume, (0, 1))
        crop_raw_volume = np.tile(crop_raw_volume, (1, 3, 1, 1, 1))

        logits =  ONNX_inference_from_session(crop_raw_volume, onnx_session)
        # TODO: working on batch case
        logits = logits[0]
        pred_prob = np.exp(logits) / np.sum(np.exp(logits))
        if pred_prob[0, 1] < 0.5:
            pred_volume_category[pred_volume_category==nodule_id] = 0
    return pred_volume_category
    

def error_check(onnx_pred, torch_pred):
    np.testing.assert_allclose(onnx_pred, torch_pred, rtol=1e-03, atol=1e-5)


def unpad(inputs, pad):
    unpad_slices = []
    in_shape = inputs.shape
    for dim, pad_in_dim in enumerate(pad):
        if pad_in_dim[1] == 0:
            unpad_slices.append(slice(pad_in_dim[0], in_shape[dim]))
        else:
            unpad_slices.append(slice(pad_in_dim[0], -pad_in_dim[1]))
    inputs = inputs[tuple(unpad_slices)]
    return inputs


def recover_lung_box(post_pred, lung_box, input_shape):
    input_arr = np.zeros(input_shape)
    input_arr[(
        slice(lung_box[0][0], lung_box[0][1]),
        slice(lung_box[1][0], lung_box[1][1]),
        slice(lung_box[2][0], lung_box[2][1]),
    )] = post_pred
    return input_arr


def resample_back(infrence_result, old_spacing, new_spacing):
    # TODO: better interpolation
    post_result, resample_spacing = resample2(
        infrence_result, old_spacing, new_spacing, mode='nearest')
    # post_result = np.where(post_result>0.5, 1, 0)
    return post_result


def save_seg_nrrd(filename, ct_scan, direction, spacing, origin):
    ct_scan = np.transpose(ct_scan, (2, 1, 0))
    origin = origin[::-1]
    spacing = spacing[::-1]
    seg_nrrd_write(filename, ct_scan, direction, spacing, origin)


def main():
    input_file = args.input
    nodulenet = NoduleNet(config)
    exe_path = os.path.dirname(sys.executable)
    model_dir = args.model_dir if args.model_dir is not None else os.path.join(exe_path, 'model')
    pred_dir = args.pred_dir if args.pred_dir is not None else os.path.join(exe_path, 'predict')
    os.makedirs(pred_dir, exist_ok=True)

    feature_net_session = onnxruntime.InferenceSession(
        os.path.join(model_dir, "f.onnx"))
    rpn_head_session = onnxruntime.InferenceSession(
        os.path.join(model_dir, "rp.onnx"))
    rcnn_head_session = onnxruntime.InferenceSession(
        os.path.join(model_dir, "rc.onnx"))
    mask_head_session = onnxruntime.InferenceSession(
        os.path.join(model_dir, "m.onnx"))
    if os.path.exists(os.path.join(model_dir, "cls.onnx")):
        nodule_cls_session = onnxruntime.InferenceSession(
            os.path.join(model_dir, "cls.onnx"))
    else:
        nodule_cls_session = None
    rcnn_crop = nodulenet.rcnn_crop

    input_image, origin, spacing, direction = load_itk_image(input_file)

    preprocess_result = preprocess_op_new(input_image, spacing)
    _, lung_box, preprocess_input, shape_before_lung_box, preprocess_time = preprocess_result
    p_time = sum(list(preprocess_time.values()))
    preprocess_input, pad = pad2factor(preprocess_input)
    preprocess_input = preprocess_input[None, None]
    preprocess_input = (preprocess_input.astype(np.float32) - 128.) / 128.

    filename = os.path.split(input_file)[1][:-4]
    onnx_pred, onnx_time = model_inference(
        config, preprocess_input, feature_net_session, rpn_head_session, 
        rcnn_head_session, mask_head_session, rcnn_crop, filename
    )

    # Nodule classification
    if nodule_cls_session is not None:
        onnx_pred, cls_time = nodule_cls(preprocess_input[0, 0], onnx_pred, nodule_cls_session)

    # Post processing
    @timer_func
    def post_process(pred):
        post_pred = unpad(pred, pad)
        post_pred = recover_lung_box(post_pred, lung_box, shape_before_lung_box)
        final_pred = resample_back(post_pred, np.ones(3, np.float), spacing)
        return final_pred
    final_pred, post_time = post_process(onnx_pred)

    # Output segmentation result
    # if not nodule_cls_session:
    #     filename = f'{filename}_no_cls'
    save_seg_nrrd(os.path.join(pred_dir, filename), final_pred, direction, spacing, origin)


if __name__ == '__main__':
    args, unparsed = parser.parse_known_args()
    main()
    