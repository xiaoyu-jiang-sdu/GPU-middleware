from typing import Optional, Dict, List

"""
将env与cwd组合成完整的命令
"""


def build_full_command(cmd: str,
                       source: Optional[List[str]] = None,
                       env: Optional[Dict[str, List[str]]] = None,
                       cwd: Optional[str] = None):
    parts = []

    if source:
        for sc in source:
            parts.append(f'source {sc}')

    if env:
        for k, v in env.items():
            if isinstance(v, (list, tuple)):
                value = ":".join(v)
            else:
                value = str(v)

            # 追加env
            parts.append(f'export {k}=${k}:{value}')

    if cwd:
        parts.append(f'mkdir -p "{cwd}"')
        parts.append(f'cd "{cwd}"')

    parts.append(cmd)
    return "; ".join(parts)
