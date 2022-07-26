import copy
import os

import torch
import onnxruntime
import matplotlib.pyplot as plt
import glob

from model2.nodulenet.layer import *
from model2.nodulenet.config import config
# from model2.nodulenet.config import net_config as config
from model2.nodulenet.utils.util import center_box_to_coord_box, ext2factor, clip_boxes, crop_boxes2mask_single, pad2factor
from model2.nodulenet.nodule_net import crop_mask_regions, NoduleNet
from model2.nodulenet.utils.LIDC.preprocess_TMH import load_itk_image, preprocess_op
from ONNX import ONNX_inference_from_session


def model_inference(
    cfg, inputs, feature_net_session, rpn_session, rcnn_session, mask_session, rcnn_crop, filename):
    mode = 'eval'
    use_rcnn = True
    use_mask = True

    # Image feature
    # features, feat_4 = feature_net(inputs); #print('fs[-1] ', fs[-1].shape)
    features = ONNX_inference_from_session(inputs, feature_net_session)
    # ort_inputs = {
    #     ort_session.get_inputs()[0].name: inputs,
    # }
    # # ort_inputs = {ort_session.get_inputs()[0].name: inputs}
    # features, feat_4 = ort_session.run(None, ort_inputs)

    fs = features[-2]
    feat_4 = features[-1]

    # RPN proposals
    # rpn_logits_flat, rpn_deltas_flat = rpn(fs)
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
        # print 'length of rpn proposals', rpn_proposals.shape

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
                rcnn_logits, rcnn_deltas
            ) 

        # pred_mask = np.zeros(list(inputs.shape[2:]))
        if use_mask and len(detections):
            # keep batch index, z, y, x, d, h, w, class
            if len(detections):
                crop_boxes = detections[:, [0, 2, 3, 4, 5, 6, 7, 8]].cpu().numpy().copy()
                crop_boxes[:, 1:-1] = center_box_to_coord_box(crop_boxes[:, 1:-1])
                crop_boxes = crop_boxes.astype(np.int32)
                crop_boxes[:, 1:-1] = ext2factor(crop_boxes[:, 1:-1], 4)
                crop_boxes[:, 1:-1] = clip_boxes(crop_boxes[:, 1:-1], inputs.shape[2:])
            
            # mask_probs = ONNX_inference_from_session(features, mask_session)
            # a = mask_session.get_inputs()
            mask_probs = []
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
                m = torch.zeros(features[0][0, 0].shape)
                m[z_start:z_end, y_start:y_end, x_start:x_end] = mask_prob

                # TODO: cuda
                mask_probs.append(torch.where(m.cuda()))
            mask_keep = mask_nms(cfg, mode, mask_probs, crop_boxes, inputs)
            crop_boxes = crop_boxes[mask_keep]
            detections = detections[mask_keep]
            # mask_probs = mask_probs[mask_keep]
            out_masks = []
            for keep_idx in mask_keep:
                out_masks.append(mask_probs[keep_idx])
            mask_probs = out_masks
            
            pred_mask = crop_mask_regions(mask_probs, crop_boxes, features[0].shape)

    pred_mask = pred_mask.numpy()
    save_dir = f'plot/final/{filename}'
    os.makedirs(save_dir, exist_ok=True)
    for idx, p in enumerate(pred_mask[0, 0]):
        if np.sum(p):
            print(idx)
            fig, ax = plt.subplots(1, 1)
            ax.imshow(p)
            ax.set_title('onnx output')
            fig.savefig(os.path.join(save_dir, f'{filename}_{idx}.png'))
    return pred_mask


def main():
    # f = rf'C:\Users\test\Desktop\Leon\Datasets\TMH_Nodule-preprocess\merge_wrong\TMH0003\raw\38467158349469692405660363178115017.mhd'
    # lung_f = '38467158349469692405660363178115017'
    f_list = glob.glob(rf'D:\Leon\Datasets\TMH-preprocess\preprocess_old\*_clean.nrrd')
    for f in f_list:
        input_image, origin, spacing = load_itk_image(f)
        # input_image, _, _ = preprocess_op(input_image, spacing, lung_f)
        input_image, pad = pad2factor(input_image)

        input_image = input_image[None, None]
        input_image = (input_image.astype(np.float32) - 128.) / 128.

        # # TODO: Numpy IO for testing (temporally)
        # # np.save('38467158349469692405660363178115017_pre.npy', input_image)
        # input_image = np.load('38467158349469692405660363178115017_pre.npy')
        # # ---

        nodulenet = NoduleNet(config)
        feature_net_session = onnxruntime.InferenceSession("feature_net.onnx")
        rpn_head_session = onnxruntime.InferenceSession("rpn_head.onnx")
        rcnn_head_session = onnxruntime.InferenceSession("rcnn_head.onnx")
        mask_head_session = onnxruntime.InferenceSession("mask_head.onnx")
        rcnn_crop = nodulenet.rcnn_crop
        filename = os.path.split(f)[1][:-4]
        model_inference(config, input_image, feature_net_session, rpn_head_session, 
                        rcnn_head_session, mask_head_session, rcnn_crop, filename)

                        
if __name__ == '__main__':
    main()
    