from fileinput import filename
import os
import glob
from random import sample
from pathlib import Path

import onnx
from onnx_tf.backend import prepare
import onnxruntime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import torchaudio
# from onnx_model import ONNX_inference



def ONNX_inference(inputs, onnx_model):
    """AI is creating summary for ONNX_inference

    Args:
        inputs ([type]): [description]
        onnx_model ([type]): [description]

    Returns:
        [type]: [description]
    """
    ort_session = onnxruntime.InferenceSession(onnx_model)
    # compute ONNX Runtime output prediction
    input_names = ort_session.get_inputs()
    assert len(inputs) == len(input_names)
    ort_inputs = {
        input_session.name: input_data for input_session, input_data in zip(input_names, inputs)}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs


def onnx_to_tf(onnx_path, tf_path):
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_path)
    test_onnx_to_tf(onnx_path, tf_path)


def test_onnx_to_tf(onnx_path, tf_path):
    # Define input
    # inputs = np.ones((1, 3, 128, 59), dtype=np.float32)
    # inputs = np.ones((1, 32000), dtype=np.float32)
    inputs = np.float32(np.random.rand(1, 32000))

    # Get ONNX output
    output_node = ONNX_inference([inputs], onnx_path)
    onnx_output = output_node[0]

    # Get TF output
    new_model = tf.saved_model.load(tf_path)
    infer = new_model.signatures["serving_default"]
    output = infer(tf.constant(inputs))
    tf_output = output['output'].numpy()

    # Check error
    error = check_error(onnx_output, tf_output)


def tf_to_tflite(tf_path, tflite_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)

    # # TFlite OPs
    converter.target_spec.suppoted_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    # # # TFlite OPs + Tensorflow OPs
    # # converter.target_spec.supported_ops = [
    # #     tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

    tf_lite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tf_lite_model)

    # Define input
    # TODO: Set shape automatically? or define outside
    # inputs = np.ones((1, 32000), dtype=np.float32)
    inputs = np.float32(np.random.rand(1, 32000))
    # inputs = np.ones((1, 3, 128, 59), dtype=np.float32)
    (tf_output, tflite_output) = test_tf_to_tflite(inputs, tf_path, tflite_path)
    error = check_error(tf_output, tflite_output)
    return error
    

def test_tf_to_tflite(inputs, tf_path, tflite_path):
    # Get TF output
    new_model = tf.saved_model.load(tf_path)
    infer = new_model.signatures["serving_default"]
    output = infer(tf.constant(inputs))
    tf_output = output['output'].numpy()

    # Get TF-lite output
    interpreter = build_tflite(tflite_path, inputs.shape)
    # tflite_output = tflite_inference2(inputs, interpreter)
    tflite_output = tflite_inference(inputs, interpreter)
    return (tf_output, tflite_output)


def check_error(output1, output2):
    error = output1 - output2
    print(error.max(), error.min(), np.abs(error).sum())
    np.testing.assert_allclose(output1, output2, rtol=1e-03, atol=1e-5)
    return error


# def onnx_to_tflite():
#     onnx_path = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_018\snoring.onnx'
#     _dir, filename = os.path.split(onnx_path)
#     tf_path = os.path.join(_dir, filename.split('.')[0])
#     tflite_path = os.path.join(_dir, filename.replace('.onnx', '.tflite'))

#     onnx_model = onnx.load(onnx_path)
#     tf_rep = prepare(onnx_model)
#     tf_rep.export_graph(tf_path)

#     input_nodes = tf_rep.inputs
#     output_nodes = tf_rep.outputs
#     converter = tf.lite.TFLiteConverter.from_frozen_graph(
#         tf_path, input_arrays=input_nodes, output_arrays=output_nodes)
#     converter.target_spec.suppoted_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]
#     tflite_rep = converter.convert()
#     open(tflite_path, "wb").write(tflite_rep)


def tflite_inference(inputs, interpreter):
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    # input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], tf.constant(inputs))
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


def tflite_inference2(inputs, interpreter):
    # Get input and output tensors.
    a = interpreter.get_signature_list()
    classify_lite = interpreter.get_signature_runner('serving_default')
    output_data = classify_lite(input=inputs)['output']
    return output_data

    
def build_tflite(tflite_path, input_shape):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.resize_tensor_input(0, input_shape, strict=True)
    interpreter.allocate_tensors()
    return interpreter


def snoring_tf_tflite_testing(tf_path, tflite_path):
    # inputs = np.ones((1, 3, 128, 59), dtype=np.float32)
    data_dir = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_cpp\2_21_2s_my2\csv\test'
    files = glob.glob(os.path.join(data_dir, '*.csv'))
    rand_files = sample(files, 50)

    data_dir = r'C:\Users\test\Desktop\Leon\Datasets\test\web_snoring_pre\pixel_0908_2\data'
    files = glob.glob(os.path.join(data_dir, '*.wav'))
    rand_files = sample(files, 50)

    total_tf, total_tflite = [], []
    for idx, f in enumerate(rand_files):
        if idx > 20: break
        # print(idx, f)
        # # data = np.load(f)
        # df = pd.read_csv(f, header=None)
        # data = df.to_numpy()
        # data = data.T
        # XXX
        # data = np.ones((1, 32000), np.float32)
        # data = 2*np.float32(np.random.rand(1, 32000))-1
        # data = np.float32(np.tile(data[None, None], (1, 3, 1, 1)))

        data, sr = torchaudio.load(f, normalize=False)
        data = np.float32(data.detach().cpu().numpy())
        filename = Path(f).stem

        (tf_output, tflite_output) = test_tf_to_tflite(data, tf_path, tflite_path)
        print((filename, tf_output, tflite_output, tf_output-tflite_output))
        total_tf.append(tf_output)
        total_tflite.append(tflite_output)

    total_tf = np.concatenate(total_tf, axis=0)
    total_tflite = np.concatenate(total_tflite, axis=0)
    error = np.abs(total_tf-total_tflite)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 9), dpi=1000)
    ax1.plot(total_tf[:,0], 'dodgerblue', label='tf')
    ax1.plot(total_tflite[:,0], 'limegreen', label='tflite')
    ax1.set_title('Non-snoring probability')
    # ax1.set_xlabel('sample')
    ax1.set_ylabel('value')
    ax1.legend()

    ax2.plot(total_tf[:,1], 'dodgerblue', label='tf')
    ax2.plot(total_tflite[:,1], 'limegreen', label='tflite')
    ax2.set_title('Snoring probability')
    # ax2.set_xlabel('sample')
    ax2.set_ylabel('value')
    ax2.legend()

    ax3.plot(error[:,0], 'lightcoral', label='absolute difference (non-snoring)')
    ax3.plot(error[:,1], 'mediumorchid', label='absolute difference (snoring)')
    ax3.set_title('Absolute difference')
    ax3.set_xlabel('sample')
    ax3.set_ylabel('value')
    ax3.legend()
    fig.savefig('tf_tflite_valid_value.png')



def main():
    # onnx_path = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_516\pann_MobileNetV2_run_516.onnx'
    onnx_path = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_741\pann.ResNet38_run_741.onnx'
    onnx_path = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_727\pann.MobileNetV2_run_727.onnx'
    onnx_path = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_777\pann.MobileNetV2_run_777.onnx'
    onnx_path = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_813\pann.MobileNetV2_run_813.onnx'
    # onnx_path = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_780\pann.ResNet38_run_780.onnx'
    # onnx_path = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_723\pann.MobileNetV2_run_723.onnx'
    # onnx_path = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_722\pann.ResNet54_run_722.onnx'
    # onnx_path = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_082 \snoring.onnx'
    # onnx_path = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_050\tf-2.9.0\snoring.onnx'
    onnx_path = Path(onnx_path)

    # _dir, filename = onnx_path.parent, onnx_path.stem
    tf_path = str(onnx_path.with_suffix(''))
    tflite_path = str(onnx_path.with_suffix('.tflite'))

    # onnx_to_tf(str(onnx_path), tf_path)
    tf_to_tflite(tf_path, tflite_path)
    snoring_tf_tflite_testing(tf_path, tflite_path)
    


if __name__ == '__main__':
    main()