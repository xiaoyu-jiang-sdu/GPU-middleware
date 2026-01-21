from typing import Dict, List
import yaml

from gui.data.driver.device import Device
from gui.data.driver.device_config import RuntimeConfig, DeviceConfig
from gui.data.enum.device_type import DeviceType
from gui.data.enum.ssh_type import SSHType
from gui.data.executor.local_executor import LocalExecutor
from gui.data.executor.ssh_executor import SSHExecutor
from gui.utils.ssh_utils import load_ssh_configs

"""
解析环境变量配置
"""


def parse_env_config(env_config: Dict) -> Dict[str, List[str]]:
    env_dict = {}

    if not isinstance(env_config, dict):
        return env_dict

    for key, value in env_config.items():
        if isinstance(value, list):
            env_dict[key] = [str(item) for item in value]
        elif isinstance(value, str):
            # 字符串按逗号分割
            env_dict[key] = [path.strip() for path in value.split(",") if path.strip()]
        else:
            # 其他类型转换为字符串列表
            env_dict[key] = [str(value)]
    return env_dict


"""
加载 config/device_config.yaml 下的全部设备
"""


def load_devices(device_cfgs_path, ssh_cfgs_path):
    with open(device_cfgs_path, "r") as f:
        raw = yaml.safe_load(f)

    ssh_map = load_ssh_configs(ssh_cfgs_path)
    devices = {}
    for name, cfg in raw.get("devices", {}).items():
        device_type = DeviceType(cfg["type"])
        index = cfg.get("index", 0)
        # runtime
        runtime = cfg.get("runtime", {})
        runtime_shell = runtime.get("shell", "bash")
        runtime_env = parse_env_config(runtime.get("env", {}))

        runtime_cfg = RuntimeConfig(
            shell=runtime_shell,
            source=runtime.get("source", []),
            env=runtime_env
        )

        # ssh 及 executor
        ssh = None
        if cfg["ssh"]:
            ssh = ssh_map.get(cfg["ssh"], None)
        executor = SSHExecutor(ssh) if ssh is None or ssh.type == SSHType.REMOTE else LocalExecutor()

        device_cfg = DeviceConfig(
            name=name,
            type=device_type,
            index=index,
            runtime=runtime_cfg
        )

        devices[name] = Device(
            cfg=device_cfg,
            executor=executor
        )

    return devices
