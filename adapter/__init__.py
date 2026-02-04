from .cpu import *
from .cuda import *

try:
    from .dcu import *
    DCU_AVAILABLE = True
except Exception as e:
    DCU_AVAILABLE = False
    _DCU_IMPORT_ERROR = e
    print("DCU_import failed! error:", _DCU_IMPORT_ERROR)