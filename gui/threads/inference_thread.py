import time
from PyQt5.QtCore import QThread, pyqtSignal

from gui.data.driver.device import Device
from utils.trace import recorder


class InferenceThread(QThread):
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(object, float)

    def __init__(self, model: str, device: Device):
        super().__init__()
        self.model = model
        self.device = device

    def run(self):

        start_time = time.time()
        try:
            with self.device as d:
                device_type = d.cfg.type.value
                cmd = d.cfg.runtime.py_path + " eval.py" + (f" --device={device_type} --model={self.model} "
                                                            f"--batch_size=2")
                out, err, code = d.run(cmd, cwd=d.cfg.runtime.cwd)

                # 转为字符串
                if isinstance(out, bytes):
                    out_str = out.decode("utf-8", errors="ignore")
                else:
                    out_str = str(out)

                self.log_signal.emit(out_str)
                recorder.dump()
                elapsed_ms = (time.time() - start_time) * 1000
                self.done_signal.emit(out_str, elapsed_ms)

        except Exception as e:
            self.log_signal.emit(f"[Error] 推理出错: {e}")
