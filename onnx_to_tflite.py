from fileinput import filename
import os

from onnx_tf.backend import prepare
import numpy as np
import tensorflow as tf
import onnx

from onnx_model import ONNX_inference


def onnx_to_tf(onnx_path, tf_path):
    # onnx_model = onnx.load(onnx_path)
    # tf_rep = prepare(onnx_model)
    # tf_rep.export_graph(tf_path)

    test_onnx_to_tf(onnx_path, tf_path)


def test_onnx_to_tf(onnx_path, tf_path):
    # Define input
    inputs = np.ones((1, 3, 128, 59), dtype=np.float32)

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


def check_error(output1, output2):
    error = output1 - output2
    print(error.max(), error.min(), np.abs(error).sum())
    np.testing.assert_allclose(output1, output2, rtol=1e-03, atol=1e-5)
    return error


def tf_to_tflite(tf_path, tflite_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    converter.target_spec.suppoted_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops=True
    converter.experimental_new_converter =True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tf_lite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tf_lite_model)


def onnx_to_tflite():
    onnx_path = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_018\snoring.onnx'
    _dir, filename = os.path.split(onnx_path)
    tf_path = os.path.join(_dir, filename.split('.')[0])
    tflite_path = os.path.join(_dir, filename.replace('.onnx', '.tflite'))

    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_path)

    input_nodes = tf_rep.inputs
    output_nodes = tf_rep.outputs
    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        tf_path, input_arrays=input_nodes, output_arrays=output_nodes)
    converter.target_spec.suppoted_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_rep = converter.convert()
    open(tflite_path, "wb").write(tflite_rep)


def tflite_inference(tflite_path):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)


def main():
    onnx_path = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_018\snoring.onnx'
    _dir, filename = os.path.split(onnx_path)
    tf_path = os.path.join(_dir, filename.split('.')[0])
    tflite_path = os.path.join(_dir, filename.replace('.onnx', '.tflite'))

    onnx_to_tf(onnx_path, tf_path)
    # tf_to_tflite(tf_path, tflite_path)
    # tflite_inference(tflite_path)
    
    # # new_model = tf.saved_model.load(tf_path)
    # new_model = tf.keras.models.load_model(tf_path)
    # infer = new_model.signatures["serving_default"]
    # for v in infer.trainable_variables:
    #     print(v.name)

    # MODEL_PB = os.path.join(tf_path, 'saved_model.pb')
    # from tensorflow.python.platform import gfile
    
    # # graph_def = tf.get_default_graph().as_graph_def()
    # # with gfile.FastGFile(MODEL_PB, 'rb') as f:
    # #     graph_def.ParseFromString(f.read())
    # # tf.import_graph_def(graph_def, name='')

    # onnx_to_tflite()


if __name__ == '__main__':
    main()