from typing import Optional, Tuple
import paramiko

from gui.data.driver.device_config import RuntimeConfig
from gui.data.driver.ssh_config import SSHConfig
from gui.data.executor.base import Executor
from gui.utils.cmd_utils import build_full_command


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
            stdin, stdout, stderr = self.client.exec_command(full_cmd)

            # 输出
            stdout_data = stdout.read().decode()
            stderr_data = stderr.read().decode()

            # 返回码
            return_code = stdout.channel.recv_exit_status()
            print(f"[SSH: {self.cfg.hostname}:{self.cfg.port}] return_code: {return_code}")

            return stdout_data, stderr_data, return_code

        except paramiko.SSHException as e:
            print(f"[SSH: {self.cfg.hostname}:{self.cfg.port}] SSH error executing command: {e}")
            raise
        except Exception as e:
            print(f"[SSH: {self.cfg.hostname}:{self.cfg.port}] Error executing command: {e}")
            raise
