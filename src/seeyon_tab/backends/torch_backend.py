"""Float32 PyTorch reference/CUDA backend with portable NumPy weights.

The architecture, parameter names, loss and AdamW equations are intentionally
identical to ``mlx_backend.py``. Checkpoints are handled by the common layer;
this module never unpickles an arbitrary framework object. Cross-backend
migration can preserve the explicit canonical AdamW moments and step; numerical
parity must be checked on the destination hardware. A fresh optimization phase
can instead deliberately reset moments through the common training interface.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

try:
    import torch
    from torch.nn import functional as F
except ImportError as exc:  # optional backend: core data tools do not import it
    raise ImportError(
        "The PyTorch backend requires a hardware-appropriate PyTorch installation. "
        "See docs/CUDA_MIGRATION.md."
    ) from exc


DEFAULT_OPTIMIZER = {
    "beta1": 0.9, "beta2": 0.999, "eps": 1e-8,
    "weight_decay": 0.01, "clip_norm": 1.0,
}


def _shapes(config: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    """Use the shared versioned shape contract for both frameworks."""
    from ..weights import parameter_shapes
    return parameter_shapes(dict(config))


class TorchBackend:
    """Original dense tabular ICL model and deterministic float32 optimizer.

    ``weights`` is an exact canonical mapping of float32 NumPy arrays. Each
    linear weight uses [out_features, in_features], as in the MLX backend.
    ``batch`` is a plain NumPy dictionary; ``targets`` is optional for predict
    and is never an input to the forward computation.
    """

    name = "torch"

    def __init__(self, config: Mapping[str, Any], weights: Mapping[str, np.ndarray],
                 device: str = "cpu", optimizer_config: Mapping[str, Any] | None = None):
        from ..config import model_config
        self.config = model_config(dict(config))
        d, heads = int(self.config["width"]), int(self.config["heads"])
        if d < 1 or heads < 1 or d % heads:
            raise ValueError("width must be positive and divisible by heads")
        if int(self.config["ff_multiplier"]) < 1:
            raise ValueError("ff_multiplier must be positive")
        if any(int(self.config[k]) < 0 for k in ("row_layers", "icl_layers")):
            raise ValueError("Layer counts must be nonnegative")
        if int(self.config["category_vocab"]) < 3 or int(self.config["max_features"]) < 1:
            raise ValueError("Invalid feature/category capacity")
        if self.config.get("dtype", "float32") != "float32":
            raise ValueError("This portable backend supports float32 only")
        if float(self.config.get("layer_norm_eps", 1e-5)) != 1e-5:
            raise ValueError("Portable architecture v1 fixes layer_norm_eps=1e-5")
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("PyTorch reference backend supports cpu/cuda; use MLX on Apple Silicon")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but this PyTorch runtime has no usable CUDA device")
        if self.device.type == "cuda":
            # Establish a predictable float32 baseline rather than TF32 changes
            # silently weakening the portable parity check.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.optimizer_config = dict(DEFAULT_OPTIMIZER)
        if optimizer_config:
            unknown = set(optimizer_config) - set(DEFAULT_OPTIMIZER)
            if unknown:
                raise ValueError(f"Unknown optimizer keys: {sorted(unknown)}")
            self.optimizer_config.update(optimizer_config)
        o = self.optimizer_config
        if not (0 <= float(o["beta1"]) < 1 and 0 <= float(o["beta2"]) < 1):
            raise ValueError("AdamW beta values must be in [0, 1)")
        if not all(np.isfinite(float(value)) for value in o.values()):
            raise ValueError("Optimizer settings must be finite")
        if float(o["eps"]) <= 0 or float(o["weight_decay"]) < 0 or float(o["clip_norm"]) <= 0:
            raise ValueError("eps/clip_norm must be positive and weight_decay nonnegative")
        shapes = _shapes(self.config)
        if set(weights) != set(shapes):
            raise ValueError(f"Weight names differ from architecture: missing={sorted(set(shapes)-set(weights))}; "
                             f"extra={sorted(set(weights)-set(shapes))}")
        self.weights: dict[str, torch.Tensor] = {}
        for key in sorted(shapes):
            array = np.asarray(weights[key])
            if array.shape != shapes[key] or not np.issubdtype(array.dtype, np.floating):
                raise ValueError(f"Invalid weight shape/dtype for {key}: {array.shape}/{array.dtype}")
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite weight: {key}")
            self.weights[key] = torch.tensor(array, dtype=torch.float32, device=self.device,
                                             requires_grad=True)
        self._m = {key: torch.zeros_like(value, requires_grad=False) for key, value in self.weights.items()}
        self._v = {key: torch.zeros_like(value, requires_grad=False) for key, value in self.weights.items()}
        self._step = 0

    @property
    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.weights.values())

    @staticmethod
    def attention_mask(n_support: int, n_query: int) -> np.ndarray:
        """Boolean [N,N] mask, True means blocked; matches both backends."""
        if n_support < 1 or n_query < 1:
            raise ValueError("Need positive support/query sizes")
        total = n_support + n_query
        mask = np.ones((total, total), dtype=bool)
        mask[:, :n_support] = False
        queries = np.arange(n_support, total)
        mask[queries, queries] = False
        return mask

    @staticmethod
    def _gelu(value: torch.Tensor) -> torch.Tensor:
        return 0.5 * value * (1.0 + torch.erf(value / math.sqrt(2.0)))

    def _linear(self, value: torch.Tensor, name: str) -> torch.Tensor:
        return F.linear(value, self.weights[f"{name}.weight"], self.weights[f"{name}.bias"])

    def _norm(self, value: torch.Tensor, name: str) -> torch.Tensor:
        # Explicit population variance avoids framework-specific norm defaults.
        center = value - value.mean(dim=-1, keepdim=True)
        variance = (center * center).mean(dim=-1, keepdim=True)
        return center * torch.rsqrt(variance + 1e-5) * self.weights[f"{name}.weight"] + self.weights[f"{name}.bias"]

    def _block(self, value: torch.Tensor, name: str,
               blocked: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, width = value.shape
        heads = int(self.config["heads"])
        head_dim = width // heads
        normalized = self._norm(value, f"{name}.norm1")

        def project(which: str) -> torch.Tensor:
            projected = self._linear(normalized, f"{name}.{which}_proj")
            return projected.reshape(batch, length, heads, head_dim).transpose(1, 2)

        query, key, val = project("q"), project("k"), project("v")
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dim)
        if blocked is not None:
            scores = scores.masked_fill(blocked[None, None, :, :], -float("inf"))
        attended = torch.softmax(scores, dim=-1) @ val
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        value = value + self._linear(attended, f"{name}.out_proj")
        ff = self._norm(value, f"{name}.norm2")
        ff = self._gelu(self._linear(ff, f"{name}.ff1"))
        return value + self._linear(ff, f"{name}.ff2")

    def _prepare(self, batch: Mapping[str, Any]) -> tuple[dict[str, torch.Tensor], str]:
        required = {"numeric", "categories", "missing", "is_categorical", "y_support", "task"}
        missing_keys = required - set(batch)
        if missing_keys:
            raise ValueError(f"Missing prepared batch keys: {sorted(missing_keys)}")
        task = batch["task"]
        if task not in ("classification", "regression"):
            raise ValueError("task must be classification or regression")
        numeric = np.asarray(batch["numeric"], dtype=np.float32)
        categories = np.asarray(batch["categories"])
        missing = np.asarray(batch["missing"], dtype=bool)
        is_cat = np.asarray(batch["is_categorical"], dtype=bool)
        ys = np.asarray(batch["y_support"], dtype=np.float32)
        if numeric.ndim != 2 or categories.shape != numeric.shape or missing.shape != numeric.shape:
            raise ValueError("numeric/categories/missing must have matching [N,F] shapes")
        n_rows, n_features = numeric.shape
        if not 1 <= n_features <= int(self.config["max_features"]):
            raise ValueError("Invalid feature count")
        if is_cat.shape != (n_features,) or ys.ndim != 1 or not 1 <= len(ys) < n_rows:
            raise ValueError("Invalid categorical mask or support labels")
        if not np.isfinite(numeric).all() or not np.isfinite(ys).all():
            raise ValueError("Prepared values and support labels must be finite")
        if not np.issubdtype(categories.dtype, np.integer):
            raise ValueError("Prepared category IDs must be integers")
        if categories.min() < 0 or categories.max() >= int(self.config["category_vocab"]):
            raise ValueError("Category ID outside model vocabulary")
        if task == "classification" and not np.isin(ys, [0, 1]).all():
            raise ValueError("Only binary classification is supported")
        values = {
            "numeric": torch.as_tensor(numeric, device=self.device),
            "categories": torch.as_tensor(categories, dtype=torch.long, device=self.device),
            "missing": torch.as_tensor(missing, device=self.device),
            "is_categorical": torch.as_tensor(is_cat, device=self.device),
            "y_support": torch.as_tensor(ys, device=self.device),
        }
        return values, str(task)

    def _forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        inputs, task = self._prepare(batch)
        numeric, categories = inputs["numeric"], inputs["categories"]
        is_cat, missing = inputs["is_categorical"], inputs["missing"]
        ys = inputs["y_support"]
        n_rows, n_features = numeric.shape
        n_support = len(ys)
        numeric_tokens = self._linear(self._gelu(self._linear(numeric[..., None], "numeric_in")), "numeric_out")
        category_tokens = self.weights["category_embedding.weight"][categories]
        tokens = torch.where(is_cat[None, :, None], category_tokens, numeric_tokens)
        tokens = tokens + self.weights["feature_embedding.weight"][:n_features][None, :, :]
        tokens = tokens + self.weights["type_embedding.weight"][is_cat.long()][None, :, :]
        tokens = tokens + self.weights["missing_embedding.weight"][missing.long()]
        row_tokens = self.weights["row_token"][None, None, :].expand(n_rows, 1, -1)
        row_input = torch.cat((row_tokens, tokens), dim=1)
        for i in range(int(self.config["row_layers"])):
            row_input = self._block(row_input, f"row_blocks.{i}")
        rows = row_input[:, 0, :]
        task_index = 0 if task == "classification" else 1
        if task == "classification":
            labels = self.weights["class_label_embedding.weight"][ys.long()]
        else:
            labels = self._linear(self._gelu(self._linear(ys[:, None], "reg_label_in")), "reg_label_out")
        unknown = self.weights["unknown_label"][None, :].expand(n_rows - n_support, -1)
        rows = rows + torch.cat((labels, unknown), dim=0) + self.weights["task_embedding.weight"][task_index]
        mask = torch.as_tensor(self.attention_mask(n_support, n_rows - n_support), device=self.device)
        encoded = rows[None, :, :]
        for i in range(int(self.config["icl_layers"])):
            encoded = self._block(encoded, f"icl_blocks.{i}", mask)
        queries = self._norm(encoded[0, n_support:, :], "output_norm")
        if task == "classification":
            return self._linear(queries, "class_head")
        raw = self._linear(queries, "reg_head")
        center = raw[:, 1]
        lower_gap = torch.logaddexp(torch.zeros_like(raw[:, 0]), raw[:, 0])
        upper_gap = torch.logaddexp(torch.zeros_like(raw[:, 2]), raw[:, 2])
        return torch.stack((center - lower_gap, center, center + upper_gap), dim=-1)

    def _loss(self, batch: Mapping[str, Any]) -> torch.Tensor:
        if "targets" not in batch or batch["targets"] is None:
            raise ValueError("train_step requires query targets for the loss")
        predicted = self._forward(batch)
        target_array = np.asarray(batch["targets"], dtype=np.float32)
        if target_array.shape != (len(predicted),) or not np.isfinite(target_array).all():
            raise ValueError("Targets must be finite with one value per query")
        targets = torch.as_tensor(target_array, device=self.device)
        if batch["task"] == "classification":
            if not np.isin(target_array, [0, 1]).all():
                raise ValueError("Only binary classification targets are supported")
            # Explicit logsumexp matches the MLX loss contract.
            return (torch.logsumexp(predicted, dim=-1) - predicted[torch.arange(len(targets), device=self.device), targets.long()]).mean()
        residual = targets[:, None] - predicted
        levels = predicted.new_tensor([0.1, 0.5, 0.9])
        return torch.maximum(levels * residual, (levels - 1.0) * residual).mean()

    def predict(self, batch: Mapping[str, Any]) -> np.ndarray:
        """Return logits [Q,2] or ordered quantiles [Q,3]; ignore targets."""
        with torch.no_grad():
            result = self._forward(batch)
        if not torch.isfinite(result).all():
            raise FloatingPointError("Non-finite model predictions")
        return result.detach().cpu().numpy().copy()

    def train_step(self, batches: list[Mapping[str, Any]], lr: float) -> dict[str, float | int]:
        """One accumulated AdamW update over the mean episode loss.

        Moments and decay are applied to every canonical tensor. Parameters not
        used by a particular task receive a zero gradient, matching MLX exactly.
        """
        if not batches:
            raise ValueError("Need at least one episode per optimizer update")
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("Learning rate must be positive and finite")
        for value in self.weights.values():
            value.grad = None
        total_loss = 0.0
        for batch in batches:
            loss = self._loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            (loss / len(batches)).backward()
            total_loss += float(loss.detach().cpu()) / len(batches)
        gradients = {key: value.grad if value.grad is not None else torch.zeros_like(value)
                     for key, value in self.weights.items()}
        norm = torch.sqrt(sum(torch.sum(gradient * gradient) for gradient in gradients.values()))
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite gradient norm")
        clip = float(self.optimizer_config["clip_norm"])
        factor = torch.minimum(norm.new_tensor(1.0), norm.new_tensor(clip) / (norm + 1e-6))
        beta1, beta2 = float(self.optimizer_config["beta1"]), float(self.optimizer_config["beta2"])
        eps, decay = float(self.optimizer_config["eps"]), float(self.optimizer_config["weight_decay"])
        self._step += 1
        with torch.no_grad():
            for key, value in self.weights.items():
                gradient = gradients[key] * factor
                self._m[key].mul_(beta1).add_(gradient, alpha=1.0-beta1)
                self._v[key].mul_(beta2).addcmul_(gradient, gradient, value=1.0-beta2)
                m_hat = self._m[key] / (1.0 - beta1 ** self._step)
                v_hat = self._v[key] / (1.0 - beta2 ** self._step)
                value.mul_(1.0 - lr * decay).addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)
                if not torch.isfinite(value).all():
                    raise FloatingPointError(f"Non-finite updated weight: {key}")
                value.grad = None
        self.synchronize()
        return {"loss": total_loss, "grad_norm": float(norm.detach().cpu()), "step": self._step}

    def export_weights(self) -> dict[str, np.ndarray]:
        return {key: value.detach().cpu().numpy().copy() for key, value in self.weights.items()}

    def get_optimizer_state(self) -> dict[str, np.ndarray]:
        result = {"step": np.asarray(self._step, dtype=np.int64)}
        result.update({f"m::{key}": value.cpu().numpy().copy() for key, value in self._m.items()})
        result.update({f"v::{key}": value.cpu().numpy().copy() for key, value in self._v.items()})
        return result

    def set_optimizer_state(self, state: Mapping[str, np.ndarray]) -> None:
        expected = {"step"} | {f"{prefix}::{key}" for prefix in ("m", "v") for key in self.weights}
        if set(state) != expected:
            raise ValueError("Optimizer state keys do not match this model")
        step = np.asarray(state["step"])
        if step.shape != () or not np.issubdtype(step.dtype, np.integer) or int(step) < 0:
            raise ValueError("Optimizer step must be a nonnegative integer scalar")
        converted: dict[str, dict[str, torch.Tensor]] = {"m": {}, "v": {}}
        for prefix in ("m", "v"):
            for key, weight in self.weights.items():
                array = np.asarray(state[f"{prefix}::{key}"])
                if array.shape != tuple(weight.shape) or not np.issubdtype(array.dtype, np.floating):
                    raise ValueError(f"Invalid optimizer tensor: {prefix}::{key}")
                if not np.isfinite(array).all() or (prefix == "v" and np.any(array < 0)):
                    raise ValueError(f"Invalid optimizer values: {prefix}::{key}")
                converted[prefix][key] = torch.tensor(array, dtype=torch.float32, device=self.device)
        self._step, self._m, self._v = int(step), converted["m"], converted["v"]

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def memory_stats(self) -> dict[str, int | None]:
        if self.device.type == "cuda":
            return {
                "allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
            }
        return {"allocated_bytes": None, "peak_allocated_bytes": None,
                "reserved_bytes": None, "peak_reserved_bytes": None}

    def runtime_info(self) -> dict[str, Any]:
        return {"backend": "torch", "version": str(torch.__version__),
                "device": str(self.device), "dtype": "float32",
                "cuda_version": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU",
                "parameters": self.parameter_count}
