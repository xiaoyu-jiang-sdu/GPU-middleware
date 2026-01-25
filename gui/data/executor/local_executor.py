import os
import subprocess
import threading
from typing import Tuple

from gui.data.driver.device_config import RuntimeConfig
from gui.data.executor.base import Executor
from utils.trace import parse_trace_line


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

            print(f"[LOCAL] Executing {cmd}")
            process = subprocess.Popen(
                cmd,
                shell=True,
                cwd=cwd,
                env=run_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            stdout_buf = []
            stderr_buf = []

            def _read_stream(stream, buf):
                if stream is None:
                    return
                try:
                    for line in iter(stream.readline, ""):
                        if parse_trace_line(line):
                            continue
                        buf.append(line)
                finally:
                    stream.close()
            t_out = threading.Thread(
                target=_read_stream,
                args=(process.stdout, stdout_buf),
                daemon=True
            )
            t_err = threading.Thread(
                target=_read_stream,
                args=(process.stderr, stderr_buf),
                daemon=True
            )

            t_out.start()
            t_err.start()

            return_code = process.wait()
            t_out.join()
            t_err.join()

            print(f"[LOCAL] return_code={return_code}")

            return "".join(stdout_buf), "".join(stderr_buf), return_code
        except subprocess.TimeoutExpired:
            # 超时
            print(f"[LOCAL] Command timed out: {cmd}")
            raise
        except Exception as e:
            # 出错
            print(f"[LOCAL] Error executing command: {e}")
            raise
