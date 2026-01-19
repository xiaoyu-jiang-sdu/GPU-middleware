from .cpu import *
from .cuda import *

# 可选dcu
try:
    from .dcu import *
    DCU_AVAILABLE = True
except Exception as e:
    DCU_AVAILABLE = False
    _DCU_IMPORT_ERROR = e