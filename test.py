import torch
print(torch.version.cuda)       # PyTorch 编译时 CUDA 版本
print(torch.cuda.is_available())  # GPU 是否可用
print(torch.cuda.device_count())  # 检测到多少 GPU
