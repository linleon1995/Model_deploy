from pathlib import Path

import cv2
import mlflow
import numpy as np
import onnx
import onnxruntime as rt
from mlflow.deployments import get_deploy_client

# model


def redisai_deploy(model_uri: str):
    from mlflow.deployments import get_deploy_client
    target_uri = 'redisai'  # host = localhost, port = 6379
    redisai = get_deploy_client(target_uri)
    redisai.create_deployment('YOLO_onnx', model_uri, config={'device': 'GPU'})


def deploy(model_uri: str, built_in, model_name):
    client = get_deploy_client(built_in)
    image = get_data()

    client.create_deployment(model_name, model_uri, config={'device': 'GPU'})
    prediction_df = client.predict_deployment(model_name, image)
    # List all deployments, get details of our particular deployment
    print(client.list_deployments())
    print(client.get_deployment(model_name))
    # Update our deployment to serve a different model
    client.update_deployment(model_name, "runs:/anotherRunId/myModel")
    # Delete our deployment
    client.delete_deployment(model_name)


def log_onnx_model(onnx_path):
    onnx_model = onnx.load(onnx_path)

    with mlflow.start_run() as run:
        mlflow.onnx.log_model(onnx_model, "model")


def get_data():
    target_size = 416
    data = cv2.imread(
        r'C:\Users\test\Desktop\Leon\Projects\EHS_AI_Model\yolo_detector\image.png')
    data = np.expand_dims(data, 0)
    data = data[:, :target_size, :target_size].astype(np.float32)
    data = np.swapaxes(data, 1, 3)
    return data


def pred_onnx_model(model_uri: str):
    # load onnx model
    onnx_model = mlflow.pyfunc.load_model(model_uri=model_uri)
    print("model.type:", type(onnx_model))

    image = get_data()
    pred = onnx_model.predict(image)
    print(pred['output'].shape)


def model_registry(model_uri: str, registry_name: str):
    result = mlflow.register_model(
        model_uri,
        registry_name,
        # tags={'good_shit': 'yo'}
    )
    return result


def test_mlflow_model_ops(model_uri, model_name, tags=None):
    """This function is just to test the mlflow model ops.
    """
    # Register the model with provided model name
    model_details = mlflow.register_model(
        model_uri,
        model_name,
        tags=tags
    )

    # Update
    client.update_registered_model(
        name=model_details.name,
        description=f"The model version {model_details.version} is more accurate."
    )

    # client.update_model_version(
    #     name=model_details.name,
    #     version=2,
    #     description="Update to Ver. 2"
    # )

    client.transition_model_version_stage(
        name=model_details.name,
        version=model_details.version,
        stage='production',
    )

    client.transition_model_version_stage(
        name=model_details.name,
        version=model_details.version-1,
        stage='staging',
    )


if __name__ == '__main__':
    # model_uri = './mlruns/0/6f25d263441842a6bc54b45983a52a25/artifacts/model'

    # # onnx_path = r'yolov7_tiny_coco_iou_0.45_conf_0.25_img_size_416.onnx'
    # # log_onnx_model(onnx_path)
    # # pred_onnx_model(model_uri)
    # # redisai_deploy(model_uri)

    # # deploy(model_uri, built_in='redisai', model_name='YOLO_onnx')

    # from mlflow.tracking.client import MlflowClient
    # from pprint import pprint
    # client = MlflowClient(tracking_uri='http://192.168.1.145:5000/')
    # # model_details = model_registry(model_uri, 'onnx_ver2')
    # test_mlflow_model_ops(model_uri, 'remote_test2', tags={'level': 'solid'})

    # for rm in client.search_registered_models():
    #     rm = dict(rm)
    #     print(
    #         f'{rm["name"]} - ver.{rm["latest_versions"][-1].version} - {rm["description"]}')
    #     # pprint(dict(rm), indent=4)

    model_name = 'remote_test2'
    model_version = '7'
    stage = 'Staging'
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{stage}"
    )

    image = get_data()
    pred = model.predict(image)
    print(pred['output'].shape)
    pass
