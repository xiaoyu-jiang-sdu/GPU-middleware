import torch
import numpy as np


def compare_torch_and_backend(
    module: torch.nn.Module,
    input_tensor: torch.Tensor,
    backend="dcu",
    atol=1e-5,
    rtol=1e-5,
    name=""
):
    module.eval()

    with torch.no_grad():
        torch_out = module(input_tensor)

    # backend
    from model.model import ONNXAwareModel
    backend_model = ONNXAwareModel(
        module,
        input_shape=tuple(input_tensor.shape),
        backend=backend
    )

    with torch.no_grad():
        backend_out = backend_model(input_tensor)

    # 转 numpy
    a = torch_out.detach().cpu().numpy()
    b = backend_out.to_numpy()

    max_err = np.max(np.abs(a - b))
    mean_err = np.mean(np.abs(a - b))

    print(f"[{name}]")
    print("  shape:", a.shape)
    print("  max err :", max_err)
    print("  mean err:", mean_err)

    assert max_err < atol + rtol * np.max(np.abs(a)), \
        f"{name} failed!"

    return max_err, mean_err
