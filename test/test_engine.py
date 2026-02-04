import torch
import numpy as np
from wrapper.wrapper import ONNXModelWrapper


def compare_torch_and_backend(
    module: torch.nn.Module,
    input_tensor,
    backend="dcu",
    atol=1e-5,
    rtol=1e-5,
    name=""
):
    module.eval()

    with torch.no_grad():
        torch_out = module(*input_tensor if isinstance(input_tensor, tuple) else (input_tensor,))

    # backend
    backend_model = ONNXModelWrapper(
        module,
        dummy_inputs=input_tensor,
        backend=backend
    )

    with torch.no_grad():
        b = backend_model(input_tensor)

    # 转 numpy
    a = torch_out.detach().cpu().numpy()

    # print("expected", a)
    # print("actual", b)
    max_err = np.max(np.abs(a - b))
    mean_err = np.mean(np.abs(a - b))

    print(f"[{name}]")
    print("  shape:", a.shape)
    print("  max err :", max_err)
    print("  mean err:", mean_err)

    assert max_err < atol + rtol * np.max(np.abs(a)), \
        f"{name} failed!"

    return max_err, mean_err
