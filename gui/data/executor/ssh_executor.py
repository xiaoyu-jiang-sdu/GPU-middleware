import json
from typing import Optional, Tuple
import paramiko
import threading

from gui.data.driver.device_config import RuntimeConfig
from gui.data.driver.ssh_config import SSHConfig
from gui.data.executor.base import Executor
from gui.utils.cmd_utils import build_full_command

from utils.trace import TRACE_PREFIX, recorder


class SSHExecutor(Executor):
    def __init__(self, ssh_cfg: SSHConfig):
        super().__init__()
        self.cfg = ssh_cfg
        self.client: Optional[paramiko.SSHClient] = None

    def open(self):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            # 链接SSH client
            kwargs = self.cfg.__dict__.copy()
            kwargs.pop("type")
            client.connect(**kwargs)
            self.client = client

            print(f"[SSH: {self.cfg.hostname}:{self.cfg.port}] Executor opened")

        except paramiko.AuthenticationException:
            print(f"[SSH] Authentication failed for {self.cfg.username}@{self.cfg.hostname}")
            raise
        except paramiko.SSHException as e:
            print(f"[SSH] SSH connection failed: {e}")
            raise
        except Exception as e:
            print(f"[SSH] Connection error: {e}")
            raise

    def close(self):
        if self.client is not None:
            self.client.close()
            self.client = None
        print(f"[SSH: {self.cfg.hostname}:{self.cfg.port}] Executor closed")

    def run(self, cmd, cfg: RuntimeConfig, cwd: str) -> Tuple[str, str, int]:
        """
        执行 cmd
        返回(stdout, stderr, return_code) 标准输出、错误输出、返回码
        """
        if self.client is None:
            raise RuntimeError("SSH client not connected. Call open() first.")

        print(f"[SSH: {self.cfg.hostname}:{self.cfg.port}] Executing: {cmd}")
        shell = cfg.shell or 'bash'
        inner_cmd = build_full_command(cmd,
                                       source=cfg.source or [],
                                       env=cfg.env or {},
                                       cwd=cwd)
        full_cmd = f'{shell} -lc "{inner_cmd}"'
        try:
            # 执行命令
            stdin, stdout, stderr = self.client.exec_command(full_cmd, get_pty=True)

            stdout_buf = []
            stderr_buf = []

            def _handle_trace_line(line: str) -> bool:
                if not line.startswith(TRACE_PREFIX):
                    return False
                try:
                    payload = line[len(TRACE_PREFIX):].strip()
                    event = json.loads(payload)
                    recorder.events.append(event)
                    return True
                except Exception:
                    return False

            def _read_stream(stream, buf):
                if stream is None:
                    return
                try:
                    for line in iter(stream.readline, ""):
                        if _handle_trace_line(line):
                            continue
                        buf.append(line)
                finally:
                    stream.close()
            t_out = threading.Thread(
                target=_read_stream,
                args=(stdout, stdout_buf),
                daemon=True
            )
            t_err = threading.Thread(
                target=_read_stream,
                args=(stderr, stderr_buf),
                daemon=True
            )

            t_out.start()
            t_err.start()

            return_code = stdout.channel.recv_exit_status()
            t_out.join()
            t_err.join()

            stdout_str = "".join(stdout_buf)
            stderr_str = "".join(stderr_buf)

            print(f"[SSH {self.cfg.hostname}] return_code={return_code}")

            return stdout_str, stderr_str, return_code

        except paramiko.SSHException as e:
            print(f"[SSH: {self.cfg.hostname}:{self.cfg.port}] SSH error executing command: {e}")
            raise
        except Exception as e:
            print(f"[SSH: {self.cfg.hostname}:{self.cfg.port}] Error executing command: {e}")
            raise
