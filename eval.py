import torch

from config.project_config import ProjectConfig
from models.registry import build_model
from wrapper.wrapper import ONNXModelWrapper
import numpy as np
import argparse
from utils.trace import trace_block_emit


# 模型评估参数
def parse_args():
    parser = argparse.ArgumentParser(description="ONNXAwareModel evaluation")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet50", "vit_b_16"], help="model architecture")
    parser.add_argument("--num_classes", type=int, default=10, help="model output classes")
    parser.add_argument("--device", type=str, default="cpu", help="choose device to eval model")
    parser.add_argument("--seed", type=int, default=2026, help="random seed for np and torch")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for test")
    model_args = parser.parse_args()

    return model_args


def set_configs(seed=2026):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    import os
    os.environ["TORCH_HOME"] = "E:\\torch_cache"
    args = parse_args()
    set_configs(args.seed)
    # 原始模型
    with trace_block_emit("Creating nn model", model=args.model):
        base_model, input_shape = build_model(
            model_name=args.model,
            num_classes=args.num_classes,
        )
    # 包装
    with trace_block_emit("Creating wrapper", model=args.model, backend=args.device):
        model = ONNXModelWrapper(
            base_model,
            torch.randn(*input_shape),
            backend=args.device
        )
    x = torch.randn(args.batch_size, 3, 224, 224)

    with trace_block_emit("Total evaluation", model=args.model, backend=args.device):
        out = model(x)

    print("eval finish! out info:")
    print("shape:", out.shape)
    print("mean:", out.mean(), "max:", out.max(), "min:", out.min())

    performance_dir = ProjectConfig.gui_dir() / "performance"
    performance_dir.mkdir(parents=True, exist_ok=True)

    model_dir = performance_dir / args.model
    model_dir.mkdir(parents=True, exist_ok=True)

    batch_dir = model_dir / f"batch_size_{args.batch_size}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    save_path = batch_dir / f"{args.device}.npy"
    np.save(save_path, out)
