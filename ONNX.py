import torch
import onnx
import onnxruntime
import numpy as np
import matplotlib.pyplot as plt


def torch_to_ONNX_3d(dummy_input, model, save_filename):
    print(f'Torch version: {torch.__version__}')
    print(f'ONNX version: {onnx.__version__}')

    # set the model to inference mode
    model.eval()

    # Input to the model
    dummy_input = torch.from_numpy(dummy_input)
    torch_out = model(dummy_input)

    # # Export the model
    # torch.onnx.export(model,               # model being run
    #                 dummy_input,                         # model input (or a tuple for multiple inputs)
    #                 save_filename,   # where to save the model (can be a file or file-like object)
    #                 export_params=True,        # store the trained parameter weights inside the model file
    #                 opset_version=13,          # the ONNX version to export the model to
    #                 do_constant_folding=True,  # whether to execute constant folding for optimization
    #                 input_names = ['input'],   # the model's input names
    #                 output_names = ['output'], # the model's output names
    #                 dynamic_axes={'input' : {1: 'channel', 2 : 'depth', 3: 'height', 4: 'width'},    # variable length axes
    #                               'output' : {1: 'num_class', 2: 'depth', 3: 'height', 4: 'width'}})


    # onnx_model = onnx.load(save_filename)
    # onnx.checker.check_model(onnx_model)
    onnx_model = save_filename

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    ort_outs = ONNX_inference(to_numpy(dummy_input), onnx_model)

    # compare ONNX Runtime and PyTorch results
    error = to_numpy(torch_out) - ort_outs[0]
    
    for idx, (t_out, o_out) in enumerate(zip(to_numpy(torch_out)[0, 0], ort_outs[0][0, 0])):
        if np.sum(t_out) or np.sum(o_out):
            print(idx)
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(t_out)
            ax[1].imshow(o_out)
            ax[0].set_title('torch')
            ax[1].set_title('onnx')
            fig.savefig(f'plot/{idx}.png')

    print(error.max(), error.min(), np.abs(error).sum())

    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-5)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def ONNX_inference(inputs, onnx_model):
    ort_session = onnxruntime.InferenceSession(onnx_model)
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: inputs}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs


def ONNX_inference_from_session(inputs, ort_session):
    ort_inputs = {ort_session.get_inputs()[0].name: inputs}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs