# Continue training with PyTorch on CUDA

Use a Linux NVIDIA host with a compatible driver. Transfer the same source revision and the run directory containing the checkpoint. For disk-backed training, also transfer the exact corpus at the configured path. The online presets regenerate their episodes locally.

Create a new Python environment on the target. The command below selects the PyTorch 2.9.1 CUDA 12.8 build for a compatible host. Install the base project from the repository root and verify a forward pass on the GPU.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
seeyon-tab doctor --backend torch --device cuda
```

Resume into a fresh directory to preserve the transferred checkpoint. Both implementations use the same parameter names and tensor orientation.

```bash
seeyon-tab train --resume runs/pilot --backend torch --device cuda --output runs/cuda-pilot --max-updates 20
seeyon-tab evaluate --checkpoint runs/cuda-pilot --backend torch --device cuda --split validation --output runs/cuda-pilot/validation.json
```

On a Linux host supporting the MLX CPU extra, the parity command compares MLX CPU and PyTorch CUDA on the imported checkpoint. It performs one diagnostic update in memory.

```bash
python -m pip install -e '.[mlx-cpu]'
seeyon-tab parity --checkpoint runs/pilot --mlx-device cpu --torch-device cuda
```
