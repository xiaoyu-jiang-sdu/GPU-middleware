import torch
from torchvision.models import resnet18
from model.model import ONNXAwareModel
import numpy as np
import argparse


# 模型评估参数
def parse_args():
    parser = argparse.ArgumentParser(description="ONNXAwareModel evaluation")

    parser.add_argument("--device", type=str, default="cuda", help="choose device to eval model")
    parser.add_argument("--seed", type=int, default=2026, help="random seed for np and torch")
    args = parser.parse_args()

    return args


def set_configs(seed=2026):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    args = parse_args()
    set_configs(args.seed)
    # 原始模型
    base_model = resnet18(num_classes=10)

    # 包装
    model = ONNXAwareModel(
        base_model,
        input_shape=(1, 3, 224, 224),
        backend=args.device
    )
    # 随机数据
    x = torch.randn(1, 3, 224, 224)

    out = model(x)
    print("eval finish! out info:")
    print(out)
    print("shape:", out.shape)
    print("mean:", out.mean(), "max:", out.max(), "min:", out.min())