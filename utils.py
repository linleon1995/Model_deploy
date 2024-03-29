import time


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


def ONNX_inference_from_session(inputs, ort_session):
    ort_inputs = {ort_session.get_inputs()[0].name: inputs}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs

    