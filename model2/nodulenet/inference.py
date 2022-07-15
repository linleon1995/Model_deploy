import sys
import os
import time
import logging
import torch
import numpy as np
import SimpleITK as sitk
from model2.nodulenet.mask_reader import MaskReader
from model2.nodulenet.config import config
from model2.nodulenet.nodule_to_nrrd import raw_nrrd_write, seg_nrrd_write
from model2.nodulenet.nodule_net import NoduleNet
# from model2.nodulenet.utils.LIDC.preprocess import resample, resample2
from model2.nodulenet.utils.util import crop_boxes2mask_single, pad2factor
from model2.nodulenet.utils.LIDC.preprocess_TMH import preprocess_op, resample, resample2
this_module = sys.modules[__name__]



def inference(model_name, image_in, image_out, origin, spacing, pid, model_path):
    config['initial_checkpoint'] = model_path
    data_dir = config['preprocessed_data_dir']
    net = prepare_model(model_name, config)
    # dataset = MaskReader(data_dir, test_set_name, config, mode='eval')
    # data_iter = iter(dataset)
    # _, truth_bboxes, truth_labels, truth_masks, mask, image = next(data_iter)
    # spacing = np.array(spacing)[::-1]
    input_image, resampled_img, pad, lung_box = prepare_input(image_in, spacing, pid)
    input_image = input_image.cuda().unsqueeze(0)
    logging.debug(f'Loading file from {data_dir}')
    logging.debug(f'Image shape: {input_image.shape}')
    # print(f'Image shape: {input_image.shape} spacing {spacing}')
    # logging.debug(f'Image2 shape: {input_image2.shape}')

    with torch.no_grad():
        infer_start = time.time()
        pred_mask = net.inference(input_image)
        infer_end = time.time()
    print(f'Inference time {infer_end-infer_start}\n')
    # ori_pred_mask = convert_back_ori(ori_pred_mask)
    # print('preddd0', pred_mask.shape)

    # unpadiing
    d, h, w = pred_mask.shape
    pred_mask = pred_mask[
        :d-pad[0][1],
        :h-pad[1][1],
        :w-pad[2][1]
    ]
    # print('preddd1', pred_mask.shape)
    
    # lung box
    pred_mask_temp = np.zeros_like(resampled_img)
    pred_mask_temp[
        lung_box[0][0]:lung_box[0][1],
        lung_box[1][0]:lung_box[1][1],
        lung_box[2][0]:lung_box[2][1],
    ] = pred_mask
    pred_mask = pred_mask_temp
    # print('preddd2', pred_mask.shape)
    
    pred_mask, resampled_spacing = resample2(pred_mask, 1/spacing, order=3, mode='nearest')
    # print('preddd3', pred_mask.shape)
    preprocessed_dir = config['preprocessed_data_dir']
    direction = np.eye(3)

    resampled_img = np.transpose(resampled_img, (2, 1, 0))
    pred_mask = np.transpose(pred_mask, (2, 1, 0))
    
    return resampled_img, pred_mask


def prepare_input(image, spacing, filename):
    # preprocess
    # print('values0', image.max(), image.min(), image.shape)
    image, resampled_img, lung_box = preprocess_op(image, spacing, filename)
    # print('values1', image.max(), image.min(), image.shape)

    # To fit model downsampling size
    input_image, pad = pad2factor(image)

    input_image = np.expand_dims(input_image, 0)
    # print('values2', input_image.max(), input_image.min(), input_image.shape)
    input_image = (input_image.astype(np.float32) - 128.) / 128.
    # print('values3', input_image.max(), input_image.min(), input_image.shape)
    # print('values', input_image.max(), input_image.min())
    input_image = torch.from_numpy(input_image).float()
    return input_image, resampled_img, pad, lung_box


def prepare_model(model_name, config):
    net = getattr(this_module, model_name)(config)
    net = net.cuda()
    initial_checkpoint = config['initial_checkpoint']
    checkpoint = torch.load(initial_checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    net.set_mode('eval')
    net.use_mask = True
    net.use_rcnn = True
    return net


if __name__ == '__main__':
    test_set_name = 'split/tmh/0_val.csv'
    model_name = 'NoduleNet'
    image_out = rf'C:\Users\test\Desktop\Leon\Weekly\0701'
    result_file = inference(config['test_set_name'], model_name, None, image_out)
    # result_file = inference(test_set_name, model_name, None, )