from abc import ABC
from typing import Tuple

from gui.data.driver.device_config import RuntimeConfig

"""
cmd 执行器基类
分为本地执行器和SSH执行器
初始化参数:
 - 1. env: 环境变量
 - 2. cwd: 工作目录
"""


class Executor(ABC):
    def __init__(self):
        self.is_open = False

    def __enter__(self):
        self.is_open = True
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.is_open = False
        self.close()

    def open(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def run(self, cmd, cfg: RuntimeConfig, cwd:str) -> Tuple[str, str, int]:
        raise NotImplementedError
