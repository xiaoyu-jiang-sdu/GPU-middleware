import json
import os
import time
import threading
from contextlib import contextmanager

from config.project_config import ProjectConfig

"""
trace recorder
记录 event 时间区间，生成 Chrome Trace JSON
"""


class TraceRecorder:
    def __init__(self):
        self.events = []
        self.pid = 1
        self._lock = threading.Lock()

    @staticmethod
    def _tid():
        # 线程 id
        return threading.get_ident() % 10000

    def record(self, name, start, end, args=None):
        # 区间事件
        with self._lock:
            self.events.append({
                "name": name,
                "ph": "X",
                "ts": start * 1e6,          # 秒 -> 微秒
                "dur": (end - start) * 1e6,
                "pid": self.pid,
                "tid": self._tid(),
                "args": args or {}
            })

    def instant(self, name, args=None):
        # 瞬时事件
        with self._lock:
            self.events.append({
                "name": name,
                "ph": "i",
                "ts": time.perf_counter() * 1e6,
                "pid": self.pid,
                "tid": self._tid(),
                "args": args or {}
            })

    def clear(self):
        self.events.clear()

    def dump(self, filename="trace.json"):
        # 默认输出到 <project_root>/trace
        trace_dir = ProjectConfig.trace_dir()
        trace_dir.mkdir(exist_ok=True)

        path = trace_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"traceEvents": self.events}, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        recorder.clear()
        return path


# 全局 recorder
recorder = TraceRecorder()


def trace(name: str, **static_args):
    """
    函数级 trace
        @trace("ONNX Export", model="resnet50")
    """
    def decorator(fn):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                end = time.perf_counter()
                recorder.record(name, start, end, static_args)
        return wrapper
    return decorator


@contextmanager
def trace_block(name: str, **args):
    """
    代码块 / 循环 trace
        with trace_block("op", op=op.name):
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        recorder.record(name, start, end, args)


def trace_instant(name: str, **args):
    recorder.instant(name, args)


# 子进程 / SSH stdout trace
# 以 __TRACE__起始的作为trace日志
TRACE_PREFIX = "__TRACE__"


def emit_trace_event(event: dict):
    # 子进程打印，父进程解析
    print(TRACE_PREFIX + json.dumps(event), flush=True)


@contextmanager
def trace_block_emit(name: str, **args):
    start = time.perf_counter() * 1e6
    try:
        yield
    finally:
        end = time.perf_counter() * 1e6
        emit_trace_event({
            "name": name,
            "ph": "X",
            "ts": start,
            "dur": end - start,
            "pid": 1,
            "tid": threading.get_ident() % 10000,
            "args": args
        })


def trace_instant_emit(name: str, **args):
    emit_trace_event({
        "name": name,
        "ph": "i",
        "ts": time.perf_counter() * 1e6,
        "pid": 1,
        "tid": threading.get_ident() % 10000,
        "args": args
    })


# 解析 trace 行
def parse_trace_line(line: str):
    if not line.startswith(TRACE_PREFIX):
        return False

    payload = line[len(TRACE_PREFIX):].strip()
    try:
        event = json.loads(payload)
        recorder.events.append(event)
        return True
    except Exception:
        return False
