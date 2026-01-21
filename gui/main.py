from gui.utils.device_utils import load_devices

device_cfgs_path = "./config/device_config.yaml"
ssh_cfgs_path = "./config/ssh_config.yaml"
devices = load_devices(device_cfgs_path=device_cfgs_path, ssh_cfgs_path=ssh_cfgs_path)

with devices["dcu0"] as dcu:
    out, err, code = dcu.run(cmd="python eval.py --device='dcu'", cwd="GPU-middleware/")
    print(out)

# with devices["cuda"] as cuda:
#     out, err, code = cuda.run(cmd=[
#         r".venv\Scripts\python.exe",
#         r"eval.py"
#     ], cwd=r"D:\PythonProject\GPU-middleware")
#     print(out)
