from gui.data.driver.device_config import DeviceConfig
from gui.data.executor.base import Executor


class Device:
    def __init__(self, cfg: DeviceConfig, executor: Executor):
        self.cfg = cfg
        self.executor = executor
        self._is_managed = False

    def __enter__(self):
        if not self.executor.is_open:
            self.executor.__enter__()
            self._is_managed = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._is_managed:
            self.executor.__exit__(exc_type, exc_val, exc_tb)
            self._is_managed = False

    # 运行 str 类型的cmd
    def run(self, cmd: str, cwd: str = ""):
        return self.executor.run(cmd, cfg=self.cfg.runtime, cwd=cwd)

    # 测试executor是否能够正常执行
    def test_connection(self) -> bool:
        stdout, stderr, return_code = self.run("echo 'test'")
        return return_code == 0
