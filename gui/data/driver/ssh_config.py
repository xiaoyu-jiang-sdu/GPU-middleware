from dataclasses import dataclass
from typing import Optional
from gui.data.enum.ssh_type import SSHType


@dataclass
class SSHConfig:
    type: SSHType
    hostname: str
    username: Optional[str] = None
    password: Optional[str] = None
    key_filename: Optional[str] = None
    port: int = 22
