from pathlib import Path

import cv2
import mlflow
import numpy as np
import onnx
import onnxruntime as rt
from mlflow.deployments import get_deploy_client


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
    model_uri = './mlruns/0/6f25d263441842a6bc54b45983a52a25/artifacts/model'

    from mlflow.tracking.client import MlflowClient
    client = MlflowClient(tracking_uri='http://192.168.1.145:5000/')

    # test_mlflow_model_ops(model_uri, 'remote_test2', tags={'level': 'solid'})

    for rm in client.search_registered_models():
        rm = dict(rm)
        print(
            f'{rm["name"]} - \
              ver.{rm["latest_versions"][-1].version} - \
              {rm["description"]}'
        )

    model_name = 'remote_test2'
    model_version = '7'
    stage = 'Staging'
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{stage}"
    )

    from mlflow_pack import get_data
    image = get_data()
    pred = model.predict(image)
    print(pred['output'].shape)

    pass
