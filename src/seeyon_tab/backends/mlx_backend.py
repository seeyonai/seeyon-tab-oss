"""MLX implementation of the portable Seeyon Tab research architecture.

MLX is imported only when an adapter is instantiated. Data generation, NumPy
preprocessing and checkpoint inspection therefore work without MLX installed.

This backend intentionally uses explicit attention and float32 arithmetic. That
keeps the mathematical contract inspectable and comparable with the Torch
backend. It is a correctness-first starting point, not a tuned Metal kernel.
"""
from __future__ import annotations

import importlib.metadata
import math
from typing import Any

import numpy as np


def _load_mlx():
    try:
        import mlx.core as mx
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "MLX is unavailable. On an Apple Silicon Mac install the project "
            "with its MLX extra; for Linux CPU validation install 'mlx[cpu]'."
        ) from exc
    return mx


class MLXBackend:
    """Differentiable MLX model with portable NumPy weights and AdamW state.

    Canonical linears store [out_features, in_features] weights. A prepared
    batch contains numeric, categories, missing, is_categorical, y_support and
    task; train_step additionally requires targets. Only targets participate in
    the query loss. No query labels enter the forward computation.
    """

    name = "mlx"

    def __init__(
        self,
        config: dict[str, Any],
        weights: dict[str, np.ndarray],
        device: str = "gpu",
        optimizer_config: dict[str, Any] | None = None,
    ):
        self.mx = _load_mlx()
        from ..config import model_config
        self.config = model_config(config)
        from ..weights import parameter_shapes
        expected_shapes = parameter_shapes(self.config)
        if set(weights) != set(expected_shapes):
            raise ValueError("Weight names do not match the portable architecture")
        for key, shape in expected_shapes.items():
            value = np.asarray(weights[key])
            if value.shape != shape or not np.issubdtype(value.dtype, np.floating):
                raise ValueError(f"Invalid weight shape/dtype for {key}")
        if self.config.get("dtype", "float32") != "float32":
            raise ValueError("This portable backend supports float32 only")
        if float(self.config.get("layer_norm_eps", 1e-5)) != 1e-5:
            raise ValueError("Portable architecture v1 fixes layer_norm_eps=1e-5")
        if device not in ("cpu", "gpu", "metal"):
            raise ValueError("MLX device must be cpu, gpu or metal")
        if device in ("gpu", "metal"):
            metal_available = bool(self.mx.metal.is_available())
            cuda_module = getattr(self.mx, "cuda", None)
            cuda_available = bool(cuda_module and cuda_module.is_available())
            if device == "metal" and not metal_available:
                raise RuntimeError("Apple Metal is unavailable in this MLX runtime")
            if not (metal_available or cuda_available):
                raise RuntimeError(
                    "MLX GPU requested but neither Metal nor CUDA is available; "
                    "use device='cpu' for Linux CPU checks."
                )
            self._device = self.mx.gpu
            self.device = "gpu"
        else:
            self._device = self.mx.cpu
            self.device = "cpu"
        self.mx.set_default_device(self._device)
        self.params = {
            name: self.mx.array(np.asarray(value, dtype=np.float32))
            for name, value in sorted(weights.items())
        }
        if not self.params:
            raise ValueError("No model weights supplied")
        if not all(np.isfinite(np.asarray(value)).all() for value in weights.values()):
            raise ValueError("Model weights must be finite")
        defaults = {
            "beta1": 0.9, "beta2": 0.999, "eps": 1e-8,
            "weight_decay": 0.01, "clip_norm": 1.0,
        }
        if set(optimizer_config or {}) - set(defaults):
            raise ValueError("Unknown optimizer configuration keys")
        self.optimizer_config = {**defaults, **(optimizer_config or {})}
        oc = self.optimizer_config
        if not all(np.isfinite(float(value)) for value in oc.values()):
            raise ValueError("Optimizer configuration must be finite")
        if not (0 <= oc["beta1"] < 1 and 0 <= oc["beta2"] < 1):
            raise ValueError("AdamW betas must be in [0, 1)")
        if oc["eps"] <= 0 or oc["clip_norm"] <= 0 or oc["weight_decay"] < 0:
            raise ValueError("Invalid AdamW epsilon, clipping norm or weight decay")
        self.m = {key: self.mx.zeros_like(value) for key, value in self.params.items()}
        self.v = {key: self.mx.zeros_like(value) for key, value in self.params.items()}
        self.step = 0
        self.mx.eval(self.params, self.m, self.v)

    @property
    def parameter_count(self) -> int:
        return sum(value.size for value in self.params.values())

    def _linear(self, p, prefix, x):
        return x @ self.mx.transpose(p[prefix + ".weight"]) + p[prefix + ".bias"]

    def _gelu(self, x):
        return 0.5 * x * (1.0 + self.mx.erf(x / math.sqrt(2.0)))

    def _norm(self, p, prefix, x):
        mean = self.mx.mean(x, axis=-1, keepdims=True)
        centered = x - mean
        variance = self.mx.mean(centered * centered, axis=-1, keepdims=True)
        normalized = centered * self.mx.rsqrt(
            variance + float(self.config.get("layer_norm_eps", 1e-5))
        )
        return normalized * p[prefix + ".weight"] + p[prefix + ".bias"]

    def _attention(self, p, prefix, x, blocked_mask=None):
        mx = self.mx
        batch, length, width = x.shape
        heads = int(self.config["heads"])
        head_width = width // heads
        projections = [
            self._linear(p, prefix + "." + name, x).reshape(
                batch, length, heads, head_width
            )
            for name in ("q_proj", "k_proj", "v_proj")
        ]
        q, k, v = [mx.transpose(value, (0, 2, 1, 3)) for value in projections]
        scores = (q @ mx.swapaxes(k, -1, -2)) / math.sqrt(head_width)
        if blocked_mask is not None:
            scores = mx.where(blocked_mask[None, None, :, :], -float("inf"), scores)
        attention = mx.softmax(scores, axis=-1)
        context = mx.transpose(attention @ v, (0, 2, 1, 3)).reshape(
            batch, length, width
        )
        return self._linear(p, prefix + ".out_proj", context)

    def _block(self, p, prefix, x, blocked_mask=None):
        x = x + self._attention(
            p, prefix, self._norm(p, prefix + ".norm1", x), blocked_mask
        )
        hidden = self._linear(p, prefix + ".ff1", self._norm(p, prefix + ".norm2", x))
        return x + self._linear(p, prefix + ".ff2", self._gelu(hidden))

    def _prepare(self, batch):
        mx = self.mx
        task = batch["task"]
        if task not in ("classification", "regression"):
            raise ValueError("Only binary classification and regression are supported")
        numeric = np.asarray(batch["numeric"], dtype=np.float32)
        original_categories = np.asarray(batch["categories"])
        if not np.issubdtype(original_categories.dtype, np.integer):
            raise ValueError("Prepared category IDs must be integers")
        if original_categories.size and (
            original_categories.min() < 0
            or original_categories.max() >= self.config["category_vocab"]
        ):
            raise ValueError("Category ID exceeds the model vocabulary")
        categories = original_categories.astype(np.int32)
        missing = np.asarray(batch["missing"], dtype=bool)
        categorical = np.asarray(batch["is_categorical"], dtype=bool)
        support = np.asarray(batch["y_support"], dtype=np.float32).reshape(-1)
        if numeric.ndim != 2 or not 1 <= numeric.shape[1] <= self.config["max_features"]:
            raise ValueError("Invalid feature shape for this model")
        if categories.shape != numeric.shape or missing.shape != numeric.shape:
            raise ValueError("Prepared feature arrays must have matching shapes")
        if categorical.shape != (numeric.shape[1],):
            raise ValueError("is_categorical must have one entry per feature")
        if not 1 <= len(support) < len(numeric):
            raise ValueError("Need at least one support and query row")
        if not np.isfinite(numeric).all() or not np.isfinite(support).all():
            raise ValueError("Prepared numeric values and support labels must be finite")
        if task == "classification" and not np.isin(support, (0, 1)).all():
            raise ValueError("Classification support labels must be binary")
        n = len(numeric)
        blocked = np.ones((n, n), dtype=bool)
        blocked[:, :len(support)] = False
        qids = np.arange(len(support), n)
        blocked[qids, qids] = False
        return {
            "numeric": mx.array(numeric), "categories": mx.array(categories),
            "missing": mx.array(missing), "is_categorical": mx.array(categorical),
            "y_support": mx.array(support), "task": task,
            "blocked_mask": mx.array(blocked),
        }

    def _forward(self, p, batch):
        mx = self.mx
        numeric, categories = batch["numeric"], batch["categories"]
        missing, categorical = batch["missing"], batch["is_categorical"]
        support, task = batch["y_support"], batch["task"]
        rows, features = numeric.shape
        width = int(self.config["width"])
        n_support = support.shape[0]
        numeric_tokens = self._linear(
            p, "numeric_out", self._gelu(self._linear(p, "numeric_in", numeric[..., None]))
        )
        category_tokens = p["category_embedding.weight"][categories]
        tokens = mx.where(categorical[None, :, None], category_tokens, numeric_tokens)
        tokens = tokens + p["feature_embedding.weight"][mx.arange(features)][None, :, :]
        tokens = tokens + p["type_embedding.weight"][categorical.astype(mx.int32)][None, :, :]
        tokens = tokens + p["missing_embedding.weight"][missing.astype(mx.int32)]
        row_token = mx.broadcast_to(p["row_token"][None, None, :], (rows, 1, width))
        hidden = mx.concatenate((row_token, tokens), axis=1)
        for index in range(int(self.config["row_layers"])):
            hidden = self._block(p, f"row_blocks.{index}", hidden)
        hidden = hidden[:, 0, :]
        if task == "classification":
            labels = p["class_label_embedding.weight"][support.astype(mx.int32)]
            task_index = 0
        else:
            labels = self._linear(
                p, "reg_label_out",
                self._gelu(self._linear(p, "reg_label_in", support[:, None])),
            )
            task_index = 1
        query_labels = mx.broadcast_to(p["unknown_label"][None, :], (rows - n_support, width))
        hidden = hidden + mx.concatenate((labels, query_labels), axis=0)
        hidden = hidden + p["task_embedding.weight"][task_index]
        hidden = hidden[None, :, :]
        for index in range(int(self.config["icl_layers"])):
            hidden = self._block(
                p, f"icl_blocks.{index}", hidden, batch["blocked_mask"]
            )
        hidden = self._norm(p, "output_norm", hidden[0, n_support:, :])
        if task == "classification":
            return self._linear(p, "class_head", hidden)
        raw = self._linear(p, "reg_head", hidden)
        center = raw[:, 1]
        zero = mx.array(0.0, dtype=mx.float32)
        return mx.stack(
            (center - mx.logaddexp(raw[:, 0], zero),
             center,
             center + mx.logaddexp(raw[:, 2], zero)),
            axis=-1,
        )

    def _loss(self, predictions, targets, task):
        mx = self.mx
        if task == "classification":
            chosen = mx.take_along_axis(
                predictions, targets.astype(mx.int32)[:, None], axis=-1
            ).squeeze(-1)
            return mx.mean(mx.logsumexp(predictions, axis=-1) - chosen)
        levels = mx.array([0.1, 0.5, 0.9], dtype=mx.float32)
        residuals = targets[:, None] - predictions
        return mx.mean(mx.maximum(levels * residuals, (levels - 1.0) * residuals))

    def predict(self, batch: dict[str, Any]) -> np.ndarray:
        """Return logits or normalized quantiles; ignore any targets in batch."""
        self.mx.set_default_device(self._device)
        predicted = self._forward(self.params, self._prepare(batch))
        self.mx.eval(predicted)
        result = np.array(predicted, dtype=np.float32)
        if not np.isfinite(result).all():
            raise FloatingPointError("Non-finite MLX prediction")
        return result

    def train_step(self, batches: list[dict[str, Any]], lr: float) -> dict[str, float | int]:
        """Average gradients over episodes, then take one clipped AdamW update."""
        if not batches:
            raise ValueError("At least one episode is required")
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("Learning rate must be finite and positive")
        mx = self.mx
        mx.set_default_device(self._device)
        accumulated = {key: mx.zeros_like(value) for key, value in self.params.items()}
        total_loss = 0.0
        for batch in batches:
            if "targets" not in batch or batch["targets"] is None:
                raise ValueError("train_step requires query targets for the loss")
            prepared = self._prepare(batch)
            targets = np.asarray(batch["targets"], dtype=np.float32).reshape(-1)
            query_count = prepared["numeric"].shape[0] - prepared["y_support"].shape[0]
            if targets.shape != (query_count,) or not np.isfinite(targets).all():
                raise ValueError("Training query targets must be finite and match query rows")
            if prepared["task"] == "classification" and not np.isin(targets, (0, 1)).all():
                raise ValueError("Classification targets must be binary")
            y = mx.array(targets)

            def loss_fn(params):
                return self._loss(self._forward(params, prepared), y, prepared["task"])

            loss, gradients = mx.value_and_grad(loss_fn)(self.params)
            mx.eval(loss, gradients)
            loss_value = float(loss.item())
            if not math.isfinite(loss_value):
                raise FloatingPointError("Non-finite MLX training loss")
            total_loss += loss_value / len(batches)
            for key in accumulated:
                accumulated[key] = accumulated[key] + gradients[key] / len(batches)
        norm = mx.sqrt(sum(mx.sum(g * g) for g in accumulated.values()))
        mx.eval(norm, accumulated)
        grad_norm = float(norm.item())
        if not math.isfinite(grad_norm):
            raise FloatingPointError("Non-finite MLX gradient norm")
        oc = self.optimizer_config
        clip_scale = min(1.0, float(oc["clip_norm"]) / (grad_norm + 1e-6))
        b1, b2, eps = float(oc["beta1"]), float(oc["beta2"]), float(oc["eps"])
        self.step += 1
        for key in self.params:
            gradient = accumulated[key] * clip_scale
            self.m[key] = b1 * self.m[key] + (1.0 - b1) * gradient
            self.v[key] = b2 * self.v[key] + (1.0 - b2) * gradient * gradient
            corrected_m = self.m[key] / (1.0 - b1 ** self.step)
            corrected_v = self.v[key] / (1.0 - b2 ** self.step)
            self.params[key] = (
                self.params[key] * (1.0 - lr * float(oc["weight_decay"]))
                - lr * corrected_m / (mx.sqrt(corrected_v) + eps)
            )
        mx.eval(self.params, self.m, self.v)
        return {"loss": total_loss, "grad_norm": grad_norm, "step": self.step}

    def export_weights(self) -> dict[str, np.ndarray]:
        self.mx.eval(self.params)
        return {key: np.array(value, dtype=np.float32) for key, value in self.params.items()}

    def get_optimizer_state(self) -> dict[str, np.ndarray]:
        """Portable first/second moments plus global step, without pickle."""
        self.mx.eval(self.m, self.v)
        return {
            "step": np.asarray(self.step, dtype=np.int64),
            **{f"m::{key}": np.array(value, dtype=np.float32) for key, value in self.m.items()},
            **{f"v::{key}": np.array(value, dtype=np.float32) for key, value in self.v.items()},
        }

    def set_optimizer_state(self, state: dict[str, np.ndarray]) -> None:
        """Restore moments; optimizer hyperparameters come from run config."""
        expected = {"step"} | {f"{prefix}::{key}" for key in self.params for prefix in ("m", "v")}
        if set(state) != expected:
            raise ValueError("Optimizer state keys do not match model weights")
        step_array = np.asarray(state["step"])
        if step_array.shape != () or not np.issubdtype(step_array.dtype, np.integer):
            raise ValueError("Optimizer step must be an integer scalar")
        step = int(step_array)
        if step < 0:
            raise ValueError("Optimizer step must be nonnegative")
        m, v = {}, {}
        for key, parameter in self.params.items():
            for prefix, dest in (("m", m), ("v", v)):
                value = np.asarray(state[f"{prefix}::{key}"])
                if (value.shape != parameter.shape or not np.issubdtype(value.dtype, np.floating)
                        or not np.isfinite(value).all()):
                    raise ValueError(f"Invalid optimizer moment for {key}")
                if prefix == "v" and np.any(value < 0):
                    raise ValueError(f"Negative second moment for {key}")
                dest[key] = self.mx.array(value.astype(np.float32))
        self.m, self.v, self.step = m, v, step
        self.mx.eval(self.m, self.v)

    def runtime_info(self) -> dict[str, Any]:
        """Environment metadata, with memory statistics where MLX exposes them."""
        info: dict[str, Any] = {
            "backend": self.name, "version": importlib.metadata.version("mlx"),
            "device": self.device, "dtype": "float32",
            "parameters": self.parameter_count,
            "device_name": "CPU" if self.device == "cpu" else "MLX GPU",
            "metal_available": bool(self.mx.metal.is_available()),
        }
        for name in ("get_active_memory", "get_peak_memory"):
            method = getattr(self.mx, name, None)
            try:
                info[name.removeprefix("get_") + "_bytes"] = int(method()) if method else None
            except (RuntimeError, ValueError):
                info[name.removeprefix("get_") + "_bytes"] = None
        return info

    def synchronize(self) -> None:
        self.mx.eval(self.params, self.m, self.v)
        self.mx.synchronize()

    def memory_stats(self) -> dict[str, int | None]:
        def optional(name):
            try:
                method = getattr(self.mx, name, None)
                return int(method()) if method else None
            except (RuntimeError, ValueError):
                return None
        return {
            "allocated_bytes": optional("get_active_memory"),
            "peak_allocated_bytes": optional("get_peak_memory"),
            "reserved_bytes": optional("get_cache_memory"),
            "peak_reserved_bytes": None,
        }

