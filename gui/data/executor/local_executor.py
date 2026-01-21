import os
import subprocess
from typing import Tuple

from gui.data.driver.device_config import RuntimeConfig
from gui.data.executor.base import Executor


class LocalExecutor(Executor):
    def __init__(self):
        super().__init__()

    def open(self):
        print("[LOCAL] Executor opened")

    def close(self):
        print("[LOCAL] Executor closed")

    def run(self, cmd, cfg: RuntimeConfig, cwd: str) -> Tuple[str, str, int]:
        if not self.is_open:
            self.open()

        try:
            run_env = os.environ.copy()
            for k, v_list in cfg.env.items():
                value = ";".join(v_list)
                if k in run_env:
                    run_env[k] = run_env[k] + ";" + value
                else:
                    run_env[k] = value

            print(f"[LOCAL] Executing: {cmd}")
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=run_env
            )

            print(f"[LOCAL] return_code: {result.returncode}")
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            # 超时
            print(f"[LOCAL] Command timed out: {cmd}")
            raise
        except Exception as e:
            # 出错
            print(f"[LOCAL] Error executing command: {e}")
            raise
