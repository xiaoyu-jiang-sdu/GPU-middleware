from gui.utils.device_utils import load_devices

devices = load_devices()
with devices["cuda"] as d:
    out, _, _ = d.run(f".venv\Scripts\python.exe eval.py --device=cuda --model=resnet18",
                      cwd="D:\\PythonProject\\GPU-middleware")