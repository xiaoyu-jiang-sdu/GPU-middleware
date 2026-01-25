from pathlib import Path
from typing import Optional


class ProjectConfig:
    """
    全局项目配置
    - 项目根目录
    """

    _root: Optional[Path] = None

    @classmethod
    def init(cls, root: Path):
        root = root.resolve()
        if cls._root is not None:
            return
        cls._root = root

    @classmethod
    def root(cls) -> Path:
        if cls._root is None:
            raise RuntimeError(
                "ProjectConfig not initialized"
            )
        return cls._root

    # 常用派生路径
    @classmethod
    def trace_dir(cls) -> Path:
        return cls.root() / "trace"

    @classmethod
    def gui_dir(cls) -> Path:
        return cls.root() / "gui"


# 自动初始化（关键）
def _auto_init():
    # 以当前文件所在目录的父目录作为项目根目录
    root = Path(__file__).resolve().parents[1]
    ProjectConfig.init(root)


_auto_init()
