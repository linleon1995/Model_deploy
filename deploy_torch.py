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



def torch_to_ONNX_3d(dummy_input, model, save_filename):
    print(f'Torch version: {torch.__version__}')
    print(f'ONNX version: {onnx.__version__}')

    # set the model to inference mode
    model.eval()

    # Input to the model
    torch_out = model(dummy_input)

    # Export the model
    torch.onnx.export(model,               # model being run
                    dummy_input,                         # model input (or a tuple for multiple inputs)
                    save_filename,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=11,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0: 'batch_size', 2 : 'depth', 3: 'height', 4: 'width'},    # variable length axes
                                  'output' : {1: 'depth', 2: 'height', 3: 'width'}})


    onnx_model = onnx.load(save_filename)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(save_filename)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=10, atol=1e-01)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def ONNX_inference2():
    from PIL import Image
    import torchvision.transforms as transforms

    img = Image.open("cat_224x224.jpg")

    resize = transforms.Resize([224, 224])
    img = resize(img)

    img_ycbcr = img.convert('YCbCr')
    img_y, img_cb, img_cr = img_ycbcr.split()

    to_tensor = transforms.ToTensor()
    img_y = to_tensor(img_y)
    img_y.unsqueeze_(0)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    ort_session = onnxruntime.InferenceSession("super_resolution.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]

    img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

    # get the output image follow post-processing step from PyTorch implementation
    final_img = Image.merge(
        "YCbCr", [
            img_out_y,
            img_cb.resize(img_out_y.size, Image.BICUBIC),
            img_cr.resize(img_out_y.size, Image.BICUBIC),
        ]).convert("RGB")

    # Save the image, we will compare this with the output image from mobile device
    final_img.save("cat_superres_with_ort.jpg")



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

    
    # f = rf"C:\Users\test\Desktop\Leon\Projects\Nodule_Detection"
    # ckpt = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\nodule_classification\ckpt\important\run_047\4\ckpt_best.pth'
    # import sys
    # sys.path.append(f)
    # print(sys.path)
    # from postprocessing.reduce_false_positive import NoduleClassifier

    # model_builder = NoduleClassifier((32, 64, 64), checkpoint_path=ckpt, using_cuda=False)
    # torch_model = model_builder.classifier
    # torch_model.eval()
    # torch_crop = torch.from_numpy(crop)
    # loits = torch_model(torch_crop)



def torch_to_ONNX(model, model_weight, model_source='localhost'):
    print(f'Torch version: {torch.__version__}')
    print(f'ONNX version: {onnx.__version__}')

    # Load pretrained model weights
    
    batch_size = 1    # just a random number

    # Initialize model with the pretrained weights
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None

    if model_source == 'localhost':
        model.load_state_dict(model_weight, )
    elif model_source == 'url':
        model.load_state_dict(model_zoo.load_url(model_weight, map_location=map_location))

    # set the model to inference mode
    model.eval()

    # Input to the model
    x = torch.randn(batch_size, 1, 224, 224, requires_grad=False)
    torch_out = model(x)

    # Export the model
    torch.onnx.export(model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {2: 'batch_size', 3: 'batch_si'},    # variable length axes
                                  'output' : {2: 'batch_size', 3: 'batch_si'}})


    onnx_model = onnx.load("super_resolution.onnx")
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession("super_resolution.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def ONNX_inference():
    from PIL import Image
    import torchvision.transforms as transforms

    img = Image.open("cat_224x224.jpg")

    # resize = transforms.Resize([224, 224])
    resize = transforms.Resize([256, 256])
    img = resize(img)

    img_ycbcr = img.convert('YCbCr')
    img_y, img_cb, img_cr = img_ycbcr.split()

    to_tensor = transforms.ToTensor()
    img_y = to_tensor(img_y)
    img_y.unsqueeze_(0)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    ort_session = onnxruntime.InferenceSession("super_resolution.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]

    img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

    # get the output image follow post-processing step from PyTorch implementation
    final_img = Image.merge(
        "YCbCr", [
            img_out_y,
            img_cb.resize(img_out_y.size, Image.BICUBIC),
            img_cr.resize(img_out_y.size, Image.BICUBIC),
        ]).convert("RGB")

    # Save the image, we will compare this with the output image from mobile device
    final_img.save("cat_superres_with_ort.jpg")


def ONNX_to_TensorRT():
    pass

def main():
    model_weight = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
    # model_weight = rf'C:\Users\test\Desktop\Leon\Projects\Slicer_modules\nodule_extension\slicer-plugin\NvidiaAIAA\SegmentEditorNvidiaAIAALib\model\NoduleNet-model1\300.ckpt'
    model_source = 'url'

    
    class SuperResolutionNet(nn.Module):
        def __init__(self, upscale_factor, inplace=False):
            super(SuperResolutionNet, self).__init__()

            self.relu = nn.ReLU(inplace=inplace)
            self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
            self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
            self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
            self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

            self._initialize_weights()

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.pixel_shuffle(self.conv4(x))
            return x

        def _initialize_weights(self):
            init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
            init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
            init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
            init.orthogonal_(self.conv4.weight)

    # Create the super-resolution model by using the above model definition.
    torch_model = SuperResolutionNet(upscale_factor=3)

    onnx_model = torch_to_ONNX(torch_model, model_weight, model_source)


def prepare_model(net, config):
    net = net.cuda()
    initial_checkpoint = config['initial_checkpoint']
    checkpoint = torch.load(initial_checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    net.set_mode('eval')
    net.use_mask = True
    net.use_rcnn = True
    return net


def main2():
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
    dummy_input = dummy_input[64:192, 64:192, 64:192]
    # dummy_input = dummy_input[:256, :128, :256]
    dummy_input = (dummy_input.astype(np.float32) - 128.) / 128.
    print(dummy_input.min(), dummy_input.max())
    dummy_input = dummy_input[np.newaxis, np.newaxis]
    dummy_input = np.tile(dummy_input, (2, 1, 1, 1, 1))
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
        onnx_model = torch_to_ONNX_3d(dummy_input, torch_model, "nodulenet.onnx")



def main3():
    
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


if __name__ == '__main__':
    main2()
    # ONNX_inference3()