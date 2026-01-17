
DCUContext 封装了一条 HIP Stream 及其绑定的 MIOpen 和 rocBLAS 执行上下文，
用于统一管理 GPU 算子的调度与执行。