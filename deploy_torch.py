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
from ONNX import torch_to_ONNX_3d, ONNX_inference, ONNX_inference_from_session



def ONNX_inference3():
    f = rf'C:\Users\test\Desktop\Leon\Datasets\TMH_Nodule-preprocess\nodulenet\crop_old\positive\Image\11029688907433245392075633136616444_000.npy'
    crop = np.load(f)
    crop = np.tile(crop[np.newaxis, np.newaxis], (1,3,1,1,1))

    ort_session = onnxruntime.InferenceSession("nodule_cls.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: crop}
    ort_outs = ort_session.run(None, ort_inputs)

    logits = ort_outs[0]
    prob = np.exp(logits) / np.sum(np.exp(logits))
    print(logits)
    print(prob)


def prepare_model(net, config):
    net = net.cuda()
    initial_checkpoint = config['initial_checkpoint']
    checkpoint = torch.load(initial_checkpoint)
    net.load_state_dict(checkpoint['state_dict'], strict=False)
    net.set_mode('eval')
    net.use_mask = True
    net.use_rcnn = True
    return net


def NoduleNet_to_ONNX():
    # import sys
    # f = rf"C:\Users\test\Desktop\Leon\Projects\Slicer_modules\nodule_extension\slicer-plugin\NvidiaAIAA"
    # sys.path.append(f)
    # from model2.nodulenet.nodule_net import NoduleNet
    # from model2.nodulenet.config import config

    # dummy_input = torch.ones(1, 1, 64, 64, 64).cuda()
    # # dummy_input = torch.randn(1, 1, 64, 64, 64, requires_grad=True).cuda()
    # torch_model = NoduleNet(config)
    # torch_model = torch_model.cuda()
    # torch_model.use_mask = True
    # torch_model.use_rcnn = True
    # torch_model.eval()
    # with torch.no_grad():
    #     output = torch_model.inference(dummy_input)
    # np.save('test_same.npy', output)
    
    
    from model2.nodulenet.nodule_net import NoduleNet
    from model2.nodulenet.config import config
    import time
    import SimpleITK as sitk

    print(time.ctime(time.time()))
    dummy_input = torch.ones(1, 1, 64, 64, 64).cuda()
    dummy_input2 = torch.randn(1, 1, 64, 64, 64, requires_grad=True).cuda()

    itkimage = sitk.ReadImage('11029688907433245392075633136616444_clean.nrrd')
    dummy_input = sitk.GetArrayFromImage(itkimage)
    # dummy_input = dummy_input[:128, :128, :128]
    dummy_input, pad = pad2factor(dummy_input)

    # dummy_input = dummy_input[64:192, 64:193, 64:194]
    # dummy_input = dummy_input[:256, :128, :256]
    dummy_input = (dummy_input.astype(np.float32) - 128.) / 128.
    print(dummy_input.min(), dummy_input.max())
    dummy_input = dummy_input[np.newaxis, np.newaxis]
    # dummy_input = np.tile(dummy_input, (2, 1, 1, 1, 1))
    dummy_input = torch.from_numpy(dummy_input).cuda()

    print(time.ctime(time.time()))
    
    torch_model = NoduleNet(config)
    torch_model = prepare_model(torch_model, config)
    # torch_model = torch_model.cuda()
    # torch_model.use_mask = True
    # torch_model.use_rcnn = True
    # torch_model.eval()

    # with torch.no_grad():
    #     output = torch_model(dummy_input)
    # print(time.ctime(time.time()))
    # # np.save('test_same.npy', output)

    # history = np.load('test_same.npy')
    # if np.all(history == output):
    #     print('good')
    
    with torch.no_grad():
        onnx_model = torch_to_ONNX_3d(dummy_input, torch_model, "nodulenet_slicing.onnx")



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
    onnx_model = torch_to_ONNX_3d(dummy_input, torch_model, "nodule_cls_ones.onnx")


def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        # print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result, t2-t1
    return wrap_func


def main():
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
        if idx%100 == 0 and idx>0:
            print(idx)
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
    print(f'ONNX execution time {np.mean(onnx_time)} in {len(onnx_time)} cases')
    print(f'Torch execution time {np.mean(torch_total_time)} in {len(torch_total_time)} cases')
    print(f'Error {np.mean(total_error)} in {len(total_error)} cases')
    print(f'{error} for {crop.shape}')


if __name__ == '__main__':
    # NoduleNet_to_ONNX()
    # ONNX_inference3()
    main()