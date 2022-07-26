
import os
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
    model = model.cuda()

    # Input to the model
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    if isinstance(dummy_input, (list, tuple)): 
        dummy_input_t = []
        for np_input in dummy_input:
            if isinstance(np_input, (float, int)):
                inputs = torch.tensor(np_input)
            else:
                inputs = torch.from_numpy(np_input)
            inputs = inputs.cuda()
            dummy_input_t.append(inputs)
    else:
         dummy_input_t = torch.from_numpy(dummy_input)
         dummy_input_t = dummy_input_t.cuda()

    with torch.no_grad():
        if isinstance(dummy_input_t, (list, tuple)):
            torch_out_t = model(*dummy_input_t)
        else:
            torch_out_t = model(dummy_input_t)

        # TODO: better if-else statement, check this func is workable for list, tuple and tensor (input, output)
        if isinstance(torch_out_t, (list, tuple)):
            torch_out = []
            for tensor in list(torch_out_t):
                torch_out.append(to_numpy(tensor))
        else:
            torch_out = to_numpy(torch_out_t)

    # Export the model
    # dynamic_axes = {'input_1': [0, 2, 3], 'output_1': {0: 'output_1_variable_dim_0', 1: 'output_1_variable_dim_1'}}
    # model_proto_name = 'conv2d.onnx'
    # torch.onnx.export(model, x, model_proto_name, verbose=True, input_names=["input_1"], output_names=["output_1"],
    #                     example_outputs=y, dynamic_axes=dynamic_axes)
    torch.onnx.export(model,               # model being run
                      tuple(dummy_input_t),                         # model input (or a tuple for multiple inputs)
                      save_filename,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=13,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['crop_f4', 'crop_f2', 'input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={
                                    'input': {1: 'channel', 2 : 'depth', 3: 'height', 4: 'width'},
                                    'crop_f2': {1: 'channel', 2 : 'depth', 3: 'height', 4: 'width'},
                                    'crop_f4': {1: 'channel', 2 : 'depth', 3: 'height', 4: 'width'},
                                    # 'cat': {0: 'batch_size'},
                                    'output': {2: 'depth', 3: 'height', 4: 'width'}}
                                    # 'output': {1: 'num_class', 2: 'depth', 3: 'height', 4: 'width'}}
                      )


    # onnx_model = onnx.load(save_filename)
    # onnx.checker.check_model(onnx_model)
    onnx_model = save_filename

    ort_outs = ONNX_inference(dummy_input, onnx_model)

    # compare ONNX Runtime and PyTorch results
    for t, o in zip(torch_out[0,0], ort_outs[0][0,0]):
        error = t - o
        
        # np.random.seed(None)
        # dir_name = str(np.random.randint(100000))
        # save_dir = os.path.join('plot', f'{dir_name}')
        # os.makedirs(save_dir)
        # for idx, (t_out, o_out) in enumerate(zip(torch_out[0, 0], ort_outs[0, 0])):
        #     if np.sum(t_out) or np.sum(o_out):
        #         print(idx)
        #         fig, ax = plt.subplots(1, 2)
        #         ax[0].imshow(t_out)
        #         ax[1].imshow(o_out)
        #         ax[0].set_title('torch')
        #         ax[1].set_title('onnx')
        #         fig.savefig(os.path.join(save_dir, f'{idx}.png'))

        print(error.max(), error.min(), np.abs(error).sum())

        np.testing.assert_allclose(t, o, rtol=1e-03, atol=1e-5)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def ONNX_inference(inputs, onnx_model):
    ort_session = onnxruntime.InferenceSession(onnx_model)
    # compute ONNX Runtime output prediction
    input_names = ort_session.get_inputs()
    ort_inputs = {
        ort_session.get_inputs()[0].name: inputs[0],
        ort_session.get_inputs()[1].name: inputs[1],
        ort_session.get_inputs()[2].name: inputs[2],
        # 'cat': inputs[3],
        # ort_session.get_inputs()[3].name: inputs[3],
    }
    # ort_inputs = {ort_session.get_inputs()[0].name: inputs}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs


def ONNX_inference_from_session(inputs, ort_session):
    ort_inputs = {ort_session.get_inputs()[0].name: inputs}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs