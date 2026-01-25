from dataclasses import dataclass
from typing import Dict, List

from gui.data.enum.device_type import DeviceType


@dataclass
class RuntimeConfig:
    shell: str
    source: List[str]
    env: Dict[str, List[str]]
    py_path: str
    cwd: str


@dataclass
class DeviceConfig:
    name: str
    type: DeviceType
    index: int
    runtime: RuntimeConfig
