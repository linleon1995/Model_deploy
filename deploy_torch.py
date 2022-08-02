import os
import time
import onnx
import onnxruntime
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.init as init
import io
import numpy as np
from torch import nn
import torch.onnx
# import tensorrt as trt
from model2.nodulenet.utils.util import crop_boxes2mask_single, pad2factor
import glob
from ONNX import torch_to_ONNX, torch_to_ONNX_3d, torch_to_ONNX_3d_mask, ONNX_inference, ONNX_inference_from_session
from utils import timer_func

from model2.nodulenet.nodule_net import NoduleNet
from model2.nodulenet.config import config
import time
import SimpleITK as sitk


# def ONNX_inference3():
#     f = rf'C:\Users\test\Desktop\Leon\Datasets\TMH_Nodule-preprocess\nodulenet\crop_old\positive\Image\11029688907433245392075633136616444_000.npy'
#     crop = np.load(f)
#     crop = np.tile(crop[np.newaxis, np.newaxis], (1,3,1,1,1))

#     ort_session = onnxruntime.InferenceSession("nodule_cls.onnx")
#     ort_inputs = {ort_session.get_inputs()[0].name: crop}
#     ort_outs = ort_session.run(None, ort_inputs)

#     logits = ort_outs[0]
#     prob = np.exp(logits) / np.sum(np.exp(logits))
#     print(logits)
#     print(prob)


def prepare_model(net, config, use_cuda=True):
    if use_cuda:
        net.cuda()
    initial_checkpoint = config['initial_checkpoint']
    # checkpoint = torch.load(initial_checkpoint)
    # TODO: GPU to CPU
    checkpoint = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint['state_dict'], strict=True)
    net.set_mode('eval')
    net.use_mask = True
    net.use_rcnn = True


def NoduleCls_to_ONNX():
    f = rf"C:\Users\test\Desktop\Leon\Projects\Nodule_Detection"
    ckpt = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\nodule_classification\ckpt\important\run_047\4\ckpt_best.pth'
    import sys
    sys.path.append(f)
    print(sys.path)
    from postprocessing.reduce_false_positive import NoduleClassifier

    dummy_input = torch.randn(1, 3, 32, 64, 64, requires_grad=True)
    dummy_input = torch.ones(1, 3, 32, 64, 64, requires_grad=True)
    model_builder = NoduleClassifier((32, 64, 64), checkpoint_path=ckpt, using_cuda=False)
    torch_model = model_builder.classifier
    torch_model.eval()
    onnx_model = torch_to_ONNX_3d(dummy_input, torch_model, "onnx_model/cls.onnx")


def nodule_cls_main():
    # Nodule cls
    f = rf'C:\Users\test\Desktop\Leon\Datasets\TMH_Nodule-preprocess\nodulenet\crop_old\positive\Image'
    f_list = glob.glob(os.path.join(f, '*.npy'))

    ort_session = onnxruntime.InferenceSession("nodule_cls.onnx")

    ckpt = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\nodule_classification\ckpt\important\run_047\4\ckpt_best.pth'
    import sys
    cls_module_path = rf"C:\Users\test\Desktop\Leon\Projects\Nodule_Detection"
    sys.path.append(cls_module_path)
    from postprocessing.reduce_false_positive import NoduleClassifier
    model_builder = NoduleClassifier((32, 64, 64), checkpoint_path=ckpt, using_cuda=False)
    torch_model = model_builder.classifier
    torch_model.eval()

    onnx_total_time = []
    torch_total_time = []
    total_error = []
    for idx, filename in enumerate(f_list):
        # if idx%100 == 0 and idx>0:
        #     print(idx)
        crop = np.load(os.path.join(f, filename))
        crop = np.tile(crop[np.newaxis, np.newaxis], (1,3,1,1,1))
        crop_tensor = torch.from_numpy(crop)

        # torch
        @timer_func
        def torch_nodule_cls():
            torch_out = torch_model(crop_tensor)
            return torch_out

        # onnx
        @timer_func
        def onnx_nodule_cls():
            onnx_out = ONNX_inference_from_session(crop, ort_session)
            return onnx_out

        onnx_out, onnx_time = onnx_nodule_cls()
        torch_out, torch_time = torch_nodule_cls()
        onnx_total_time.append(onnx_time)
        torch_total_time.append(torch_time)
        error = np.sum(np.abs(onnx_out-torch_out.detach().numpy()))
        total_error.append(error)

    print('--')
    onnx_exc_time = np.mean(onnx_total_time)
    torch_exec_time = np.mean(torch_total_time)
    print(f'Shape {crop.shape}')
    print(f'ONNX execution time {onnx_exc_time} in {len(onnx_total_time)} cases')
    print(f'Torch execution time {torch_exec_time} in {len(torch_total_time)} cases')
    print(f'ONNX / Torch: {100*((torch_exec_time-onnx_exc_time)/torch_exec_time)} %')
    print(f'Error {np.mean(total_error)} in {len(total_error)} cases')


def nodule_det_main():
    # Nodule cls
    f = rf'D:\Leon\Datasets\TMH-preprocess\preprocess_old'
    f_list = glob.glob(os.path.join(f, '*_clean.nrrd'))
    # f_list = f_list[:5]

    ort_session = onnxruntime.InferenceSession("nodulenet_slicing.onnx")

    ckpt = '300.pt'
    import SimpleITK as sitk
    from model2.nodulenet.nodule_net import NoduleNet
    from model2.nodulenet.config import config
    torch_model = NoduleNet(config)
    prepare_model(torch_model, config, use_cuda=False)

    onnx_total_time = []
    torch_total_time = []
    total_error = []
    for idx, filename in enumerate(f_list):
        # if idx > 4: break
        if idx%1 == 0:
            print(idx)
        itkimage = sitk.ReadImage(os.path.join(f, filename))
        # itkimage = sitk.ReadImage('11029688907433245392075633136616444_clean.nrrd')
        inputs = sitk.GetArrayFromImage(itkimage)
        inputs, pad = pad2factor(inputs)
        inputs = (inputs.astype(np.float32) - 128.) / 128.
        inputs = inputs[np.newaxis, np.newaxis]
        inputs_tensor = torch.from_numpy(inputs)
        print(inputs.shape)

        # torch
        @timer_func
        def torch_nodule_det():
            torch_out = torch_model(inputs_tensor)
            return torch_out

        # onnx
        @timer_func
        def onnx_nodule_det():
            try:
                with torch.no_grad():
                    onnx_out = ONNX_inference_from_session(inputs, ort_session)
            except:
                onnx_out = None
            return onnx_out

        torch_out, torch_time = torch_nodule_det()
        onnx_out, onnx_time = onnx_nodule_det()
        if onnx_out is not None:
            torch_total_time.append(torch_time)
            onnx_total_time.append(onnx_time)
            error = np.sum(np.abs(onnx_out-torch_out.detach().numpy()))
            total_error.append(error)
            print(f'--- {filename} Error {error}')

    print('---')
    onnx_exc_time = np.mean(onnx_total_time)
    torch_exec_time = np.mean(torch_total_time)
    print(f'Shape {inputs.shape}')
    print(f'ONNX execution time {onnx_exc_time} in {len(onnx_total_time)} cases')
    print(f'Torch execution time {torch_exec_time} in {len(torch_total_time)} cases')
    print(f'ONNX / Torch: {100*((torch_exec_time-onnx_exc_time)/torch_exec_time)} %')
    print(f'Error {np.mean(total_error)} in {len(total_error)} cases')


def NoduleNet_to_ONNX_split():
    print(f'Torch version: {torch.__version__}')
    print(f'ONNX version: {onnx.__version__}')

    print(time.ctime(time.time()))
    itkimage = sitk.ReadImage('11029688907433245392075633136616444_clean.nrrd')
    dummy_input = sitk.GetArrayFromImage(itkimage)
    dummy_input, pad = pad2factor(dummy_input)

    dummy_input = (dummy_input.astype(np.float32) - 128.) / 128.
    dummy_input = dummy_input[np.newaxis, np.newaxis]
    print(dummy_input.min(), dummy_input.max())
    print(dummy_input.shape)
    # print(time.ctime(time.time()))
    
    nodulenet_model = NoduleNet(config)
    prepare_model(nodulenet_model, config, use_cuda=False)

    feature_net = nodulenet_model.feature_net
    rpn_head = nodulenet_model.rpn
    rcnn_head = nodulenet_model.rcnn_head
    mask_head = nodulenet_model.mask_head

    rpn_input = np.ones([1, 128, 92, 64, 112], dtype=np.float32)
    rcnn_input = np.ones([260, 64, 7, 7, 7], dtype=np.float32)
    mask_input = (
        np.ones([1, 128, 5, 4, 4], dtype=np.float32),
        np.ones([1, 32, 10, 8, 8], dtype=np.float32),
        np.ones([1, 1, 20, 16, 16], dtype=np.float32),
        # np.ones([1, 1, 20, 16, 16], dtype=np.float32),
        # np.ones(1, dtype=np.float32)[None],
    )
                  
    with torch.no_grad():
        feature_net = torch_to_ONNX(
            dummy_input, feature_net, "onnx_model/f.onnx", output_names=['x', 'out1', 'comb2', 'out2'],
            dynamic_axes = {
                    'input' : {0: 'batch_size', 2 : 'depth', 3: 'height', 4: 'width'},
                    'x' : {0: 'batch_size', 2: 'depth', 3: 'height', 4: 'width'},
                    'out1' : {0: 'batch_size', 2: 'depth', 3: 'height', 4: 'width'},
                    'comb2' : {0: 'batch_size', 2: 'depth', 3: 'height', 4: 'width'},
                    'out2' : {0: 'batch_size', 2: 'depth', 3: 'height', 4: 'width'}
            }
        )

        rpn_head = torch_to_ONNX(
            rpn_input, rpn_head, "onnx_model/rp.onnx", output_names=['logits', 'deltas'],
            dynamic_axes = {
                'input' : {0: 'batch_size', 2 : 'depth', 3: 'height', 4: 'width'},
                'logits' : {0: 'batch_size', 2: 'depth', 3: 'height', 4: 'width'},
                'deltas' : {0: 'batch_size', 2: 'depth', 3: 'height', 4: 'width'}
            }
        )

        rcnn_head = torch_to_ONNX(
            rcnn_input, rcnn_head, "onnx_model/rc.onnx", output_names=['logits', 'deltas'],
            dynamic_axes = {
                'input' : {0: 'batch_size', 2 : 'depth', 3: 'height', 4: 'width'},
                'logits' : {0: 'batch_size', 2: 'depth', 3: 'height', 4: 'width'},
                'deltas' : {0: 'batch_size', 2: 'depth', 3: 'height', 4: 'width'}
            }
        )

        mask_head = torch_to_ONNX(
            mask_input, mask_head, "onnx_model/m.onnx", input_names=['crop_f4', 'crop_f2', 'im'],
            dynamic_axes = {
                'crop_f4' : {0: 'batch_size', 2: 'depth', 3: 'height', 4: 'width'},
                'crop_f2' : {0: 'batch_size', 2: 'depth', 3: 'height', 4: 'width'},
                'im' : {0: 'batch_size', 2 : 'depth', 3: 'height', 4: 'width'},
                'output' : {0: 'batch_size', 2 : 'depth', 3: 'height', 4: 'width'},
            }
        )


def main():
    # nodule_det_main()
    # nodule_cls_main()
    
    # NoduleCls_to_ONNX()
    NoduleNet_to_ONNX_split()


if __name__ == '__main__':
    main()