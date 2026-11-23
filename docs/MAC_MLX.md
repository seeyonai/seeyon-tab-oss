# Training on Apple Silicon

Use a native arm64 Python. The reference environment uses Python 3.12, NumPy 2.3.5 and MLX 0.32.4. After installing the MLX extra from the repository root, verify that the requested device can execute a forward pass.

```bash
python -c 'import platform; print(platform.machine())'
seeyon-tab doctor --backend mlx --device gpu
SEEYON_TEST_MLX_DEVICE=gpu python -m unittest discover -s tests -p 'test_mlx.py' -v
```

The smoke preset exercises training and checkpoint continuation in twenty updates. The second invocation loads the optimizer and finishes the schedule.

```bash
seeyon-tab train --config configs/mac_smoke.json --output runs/smoke --max-updates 10
seeyon-tab train --resume runs/smoke --output runs/smoke
seeyon-tab evaluate --checkpoint runs/smoke --split validation --episodes 12 --output runs/smoke/validation.json
```

Install the reference PyTorch version to run all tests and compare both prediction heads across Metal and PyTorch CPU.

```bash
python -m pip install 'torch==2.9.1'
SEEYON_TEST_MLX_DEVICE=gpu python -m unittest discover -s tests -v
seeyon-tab parity --checkpoint runs/smoke --mlx-device gpu --torch-device cpu
```
