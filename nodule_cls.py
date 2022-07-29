import time
import numpy as np
from ONNX import ONNX_inference_from_session
import onnxruntime


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


def get_nodule_center(pred_volume):
    total_nodule_center = {}
    for nodule_id in np.unique(pred_volume)[1:]:
        zs, ys, xs = np.where(pred_volume==nodule_id)
        center_index, center_row, center_column = np.mean(zs), np.mean(ys), np.mean(xs)
        total_nodule_center[nodule_id] = {
            'Center': {'index': np.mean(center_index).astype('int32'), 
            'row': np.mean(center_row).astype('int32'), 
            'column': np.mean(center_column).astype('int32')}}
    return total_nodule_center
    

def crop_volume(volume, crop_range, crop_center):
    def get_interval(crop_range_dim, center, size_dim):
        begin = center - crop_range_dim//2
        end = center + crop_range_dim//2
        if begin < 0:
            begin, end = 0, end-begin
        elif end > size_dim:
            modify_distance = end - size_dim + 1
            begin, end = begin-modify_distance, size_dim-1
        # print(crop_range_dim, center, size_dim, begin, end)
        assert end-begin == crop_range_dim, \
            f'Actual cropping range {end-begin} not fit the required cropping range {crop_range_dim}'
        return (begin, end)

    index_interval = get_interval(crop_range['index'], crop_center['index'], volume.shape[0])
    row_interval = get_interval(crop_range['row'], crop_center['row'], volume.shape[1])
    column_interval = get_interval(crop_range['column'], crop_center['column'], volume.shape[2])

    return volume[index_interval[0]:index_interval[1], 
                  row_interval[0]:row_interval[1], 
                  column_interval[0]:column_interval[1]]


@timer_func
def nodule_cls(raw_volume, pred_volume_category, onnx_session):
    """AI is creating summary for nodule_cls

    Args:
        raw_volume ([D, H, W]): [description]
        pred_volume_category ([D, H, W]): [description]
        onnx_session: [description]

    Returns:
        [type]: [description]
    """
    pred_nodules = get_nodule_center(pred_volume_category)
    remove_nodule_ids = []
    crop_range = {'index': 32, 'row': 64, 'column': 64}
    for nodule_id in list(pred_nodules):
        crop_raw_volume = crop_volume(raw_volume, crop_range, pred_nodules[nodule_id]['Center'])
        crop_raw_volume = np.expand_dims(crop_raw_volume, (0, 1))
        crop_raw_volume = np.tile(crop_raw_volume, (1, 3, 1, 1, 1))

        logits =  ONNX_inference_from_session(crop_raw_volume, onnx_session)
        # TODO: working on batch case
        logits = logits[0]
        pred_prob = np.exp(logits) / np.sum(np.exp(logits))
        if pred_prob[0, 1] < 0.5:
            pred_volume_category[pred_volume_category==nodule_id] = 0
    return pred_volume_category


if __name__ == '__main__':
    # Args:
        # inputs: [D, H, W]
        # pred_mask: [D, H, W]: labelized result
        # nodule_cls_session: ONNX session
    # Return:
        # labelized result after classification
    nodule_cls_session = onnxruntime.InferenceSession("nodule_cls_ones.onnx")
    cls_pred, cls_time = nodule_cls(inputs, pred_mask, nodule_cls_session)