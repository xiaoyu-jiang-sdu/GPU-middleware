import yaml

from gui.data.driver.ssh_config import SSHConfig
from gui.data.enum.ssh_type import SSHType


def load_ssh_configs(path="./config/ssh_config.yaml"):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    ssh_map = {}
    for name, cfg in raw["ssh"].items():
        cfg["type"] = SSHType(cfg["type"])
        ssh_map[name] = SSHConfig(**cfg)

    return ssh_map
