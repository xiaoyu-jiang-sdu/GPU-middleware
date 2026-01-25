from typing import Sequence

MODEL_REGISTRY = {}


def register_model(name):
    def wrapper(fn):
        MODEL_REGISTRY[name] = fn
        return fn

    return wrapper


def build_model(model_name: str, num_classes: int):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[model_name](num_classes)


def infer_shape(weights):
    crop_size = weights.transforms().crop_size
    if isinstance(crop_size, int):
        H = W = crop_size

    elif isinstance(crop_size, Sequence):
        if len(crop_size) == 1:
            H = W = crop_size[0]
        elif len(crop_size) == 2:
            H, W = crop_size
        else:
            raise ValueError(f"Invalid crop_size: {crop_size}")

    else:
        raise TypeError(f"Unsupported crop_size type: {type(crop_size)}")

    return 1, 3, H, W
