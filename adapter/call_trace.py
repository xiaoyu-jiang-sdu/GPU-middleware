import functools
from collections import defaultdict

# op实际调用的adapter方法
ADAPTER_CALL_TRACE = defaultdict(set)


# 追踪op调用的方法
def wrap_adapter_method(method_name, fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        # 调用但是不存在op
        # 一定是executor
        op_name = getattr(self, "_current_op", "IRExecutor")
        ADAPTER_CALL_TRACE[op_name].add(method_name)
        return fn(self, *args, **kwargs)
    return wrapper
