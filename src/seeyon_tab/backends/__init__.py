"""Backends are imported only when requested; base data tools need NumPy only."""

def create_backend(name, config, weights, device, optimizer_config=None):
    if name == "mlx":
        try:
            from .mlx_backend import MLXBackend
        except ImportError as exc:
            raise RuntimeError("Install MLX for this machine; see docs/MAC_MLX.md") from exc
        return MLXBackend(config, weights, device=device, optimizer_config=optimizer_config)
    if name == "torch":
        try:
            from .torch_backend import TorchBackend
        except ImportError as exc:
            raise RuntimeError("Install a hardware-compatible PyTorch build; see docs/CUDA_MIGRATION.md") from exc
        return TorchBackend(config, weights, device=device, optimizer_config=optimizer_config)
    raise ValueError("backend must be mlx or torch")

