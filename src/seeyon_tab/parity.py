"""Exercise real MLX/PyTorch weight and AdamW-state portability.

The check compares predictions and one further update on copies of a portable
checkpoint. It never modifies the supplied checkpoint. CPU numerical agreement
does not certify Apple Metal or CUDA kernels; run this command on the intended
hardware before migrating an experiment.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint import atomic_json, load_checkpoint, resolve_checkpoint, save_checkpoint
from .config import ARCHITECTURE, validate_config
from .data import sample_episode
from .preprocess import prepare_episode
from .weights import initialize_weights, parameter_count, weights_digest

ATOL = 1e-4
RTOL = 1e-4


def _diagnostic_config(mlx_device: str) -> dict[str, Any]:
    return validate_config({
        "name": "portable-parity-diagnostic", "backend": "mlx", "device": mlx_device,
        "model": {"width": 32, "heads": 4, "row_layers": 1, "icl_layers": 2,
                  "ff_multiplier": 2, "max_features": 32, "category_vocab": 16},
        "episode": {"support": 16, "query": 8, "features": 8},
        "training": {"steps": 4, "seed": 70321, "accumulation": 1,
                     "learning_rate": 0.0002, "warmup_steps": 1},
    })


def _diagnostic_batches(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Small, fixed, common episodes; contain no held-out mechanism families."""
    support = min(32, config["episode"]["support"])
    query = min(8, config["episode"]["query"])
    features = min(16, config["episode"]["features"], config["model"]["max_features"])
    result = []
    for seed, task, family in (
        (70321, "classification", "linear"),
        (70322, "regression", "workflow_parallel"),
    ):
        raw = sample_episode(seed=seed, split="train", family=family,
                             n_support=support, n_query=query,
                             n_features=features, problem_type=task)
        result.append(prepare_episode(raw, category_vocab=config["model"]["category_vocab"]))
    return result


def _comparison(mlx_output: np.ndarray, torch_output: np.ndarray) -> dict[str, Any]:
    a, b = np.asarray(mlx_output), np.asarray(torch_output)
    same_shape = a.shape == b.shape
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not same_shape:
        return {"passed": False, "shape_mlx": list(a.shape), "shape_torch": list(b.shape),
                "finite": finite, "max_abs_difference": None, "rmse_difference": None}
    difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
    maximum = float(np.max(difference)) if finite and difference.size else None
    rmse = float(np.sqrt(np.mean(difference**2))) if finite and difference.size else None
    # Check both directions so the relative-tolerance reference is explicit and
    # symmetric. Near-zero predictions are protected by the absolute term.
    agreement = finite and np.allclose(a, b, atol=ATOL, rtol=RTOL) and np.allclose(b, a, atol=ATOL, rtol=RTOL)
    return {"passed": bool(agreement), "shape": list(a.shape), "finite": finite,
            "max_abs_difference": maximum, "rmse_difference": rmse}


def _exact_state_equal(left, right) -> bool:
    return set(left) == set(right) and all(np.array_equal(left[key], right[key]) for key in left)


def run_parity(checkpoint=None, mlx_device="cpu", torch_device="cpu", output=None) -> dict[str, Any]:
    """Check two task heads before/after a restored-moment optimizer update.

    With no checkpoint, create a tiny MLX model, perform two updates, and
    round-trip its real weights/moments through the portable checkpoint format.
    With a checkpoint, load its exact model and optimizer settings. Diagnostic
    episodes are deliberately small and the extra update is performed on
    in-memory copies, not appended to the source run.

    The diagnostic update averages two episodes and uses the configuration's
    peak learning rate, not the original run's scheduler. It validates the
    implementation of state migration, not continuation of that run's data
    cursor. The training CLI owns schedule/cursor continuation.

    Raises AssertionError if restored state differs or either task's prediction
    fails atol/rtol=1e-4, including after the diagnostic update. A requested JSON
    report is written before such a numerical assertion is raised.
    """
    from .backends.mlx_backend import MLXBackend
    from .backends.torch_backend import TorchBackend
    import torch

    torch.set_num_threads(2)
    generated_source = checkpoint is None
    with tempfile.TemporaryDirectory(prefix="seeyon-parity-") as temporary:
        if generated_source:
            config = _diagnostic_config(mlx_device)
            batches = _diagnostic_batches(config)
            initial_weights = initialize_weights(config["model"], seed=config["training"]["seed"])
            source = MLXBackend(config["model"], initial_weights, device=mlx_device,
                                optimizer_config=config["optimizer"])
            for batch in batches:
                source.train_step([batch], lr=config["training"]["learning_rate"])
            path = Path(temporary) / "source"
            save_checkpoint(path, source.export_weights(), source.get_optimizer_state(), {
                "config": config, "step": 2, "next_episode": 2,
                "purpose": "portable numerical diagnostic; no model-quality claim",
            })
            del source
        else:
            path = resolve_checkpoint(checkpoint)
            if output is not None and Path(output).resolve().is_relative_to(path.resolve()):
                raise ValueError("Write the parity report outside the source checkpoint directory")
        weights, optimizer_state, metadata = load_checkpoint(path)
        config = metadata["config"]
        batches = _diagnostic_batches(config)
        source_step = int(optimizer_state["step"])
        imported_digest = weights_digest(weights)
        mlx = MLXBackend(config["model"], weights, device=mlx_device,
                         optimizer_config=config["optimizer"])
        reference = TorchBackend(config["model"], weights, device=torch_device,
                                 optimizer_config=config["optimizer"])
        mlx.set_optimizer_state(optimizer_state)
        reference.set_optimizer_state(optimizer_state)
        restored = {
            "mlx": _exact_state_equal(optimizer_state, mlx.get_optimizer_state()),
            "torch": _exact_state_equal(optimizer_state, reference.get_optimizer_state()),
        }
        before = {batch["task"]: _comparison(mlx.predict(batch), reference.predict(batch)) for batch in batches}
        rate = float(config["training"]["learning_rate"])
        mlx_metrics = mlx.train_step(batches, lr=rate)
        torch_metrics = reference.train_step(batches, lr=rate)
        after = {batch["task"]: _comparison(mlx.predict(batch), reference.predict(batch)) for batch in batches}
        mlx_weights, torch_weights = mlx.export_weights(), reference.export_weights()
        weight_differences = {
            key: float(np.max(np.abs(mlx_weights[key].astype(np.float64)-torch_weights[key].astype(np.float64))))
            for key in weights
        }
        worst_weight = max(weight_differences, key=weight_differences.get)
        mlx_optimizer, torch_optimizer = mlx.get_optimizer_state(), reference.get_optimizer_state()
        correct_steps = int(mlx_optimizer["step"]) == int(torch_optimizer["step"]) == source_step + 1
        passed = bool(all(restored.values()) and correct_steps and
                      all(result["passed"] for phase in (before, after) for result in phase.values()))
        report = {
            "status": "passed" if passed else "failed",
            "scope": "Numerical prediction and portable-AdamW-state diagnostic, not a quality benchmark",
            "architecture": ARCHITECTURE,
            "source": "temporary MLX checkpoint after two updates" if generated_source else str(path),
            "source_checkpoint_modified": False,
            "source_weights_digest": imported_digest,
            "source_optimizer_step": source_step,
            "parameters": parameter_count(config["model"]),
            "model_config": config["model"],
            "tolerance": {"absolute": ATOL, "relative": RTOL, "symmetric_check": True},
            "runtime": {"mlx": mlx.runtime_info(), "torch": reference.runtime_info()},
            "optimizer_state_exactly_restored": restored,
            "before_update": before,
            "diagnostic_update": {
                "episodes": len(batches), "learning_rate": rate,
                "schedule_note": "Uses configured peak LR for this diagnostic only; source run is unchanged",
                "mlx": mlx_metrics, "torch": torch_metrics,
                "optimizer_step_correct": correct_steps,
            },
            "after_update": after,
            "updated_weights_max_abs_difference": weight_differences[worst_weight],
            "updated_weights_worst_parameter": worst_weight,
            "updated_weights_equal_bitwise": all(np.array_equal(mlx_weights[k], torch_weights[k]) for k in weights),
            "interpretation": (
                "Both prediction heads must agree within tolerance before and after a migrated-state update. "
                "Individual AdamW weights may differ around near-zero gradients. Neither bitwise cross-backend "
                "training nor unexecuted Metal/CUDA hardware is certified."
            ),
        }
        if output is not None:
            atomic_json(output, report)
        if not passed:
            failed = [f"{phase}/{task}" for phase, values in (("before", before), ("after", after))
                      for task, result in values.items() if not result["passed"]]
            raise AssertionError("MLX/Torch portability check failed: " + ", ".join(failed or ["optimizer restoration/step"]))
        return report
