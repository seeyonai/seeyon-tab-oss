"""Exercise actual checkpoint I/O, optimizer resume and MLX/Torch migration."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from seeyon_tab.checkpoint import load_checkpoint, save_checkpoint, sha256
from seeyon_tab.config import validate_config
from seeyon_tab.data import sample_episode
from seeyon_tab.preprocess import prepare_episode
from seeyon_tab.weights import initialize_weights

HAS_TORCH = importlib.util.find_spec("torch") is not None
try:
    import mlx.core  # noqa: F401 -- verify the optional binary runtime can load
    HAS_MLX = True
except (ImportError, OSError):
    HAS_MLX = False
if HAS_TORCH:
    import torch
    from seeyon_tab.backends.torch_backend import TorchBackend
    torch.set_num_threads(2)
if HAS_MLX:
    from seeyon_tab.backends.mlx_backend import MLXBackend


def _config(backend="torch"):
    return validate_config({
        "name": "checkpoint-test", "backend": backend, "device": "cpu",
        "model": {"width": 16, "heads": 4, "row_layers": 1, "icl_layers": 2,
                  "ff_multiplier": 2, "max_features": 16, "category_vocab": 16},
        "episode": {"support": 8, "query": 4, "features": 6},
        "training": {"steps": 6, "seed": 42, "warmup_steps": 1,
                     "learning_rate": 0.0003, "accumulation": 2},
    })


def _batches(config):
    return [prepare_episode(sample_episode(
        seed=501+i, split="train", family="linear", n_support=8, n_query=4,
        n_features=6, problem_type=task), category_vocab=config["model"]["category_vocab"])
        for i, task in enumerate(("classification", "regression"))]


def _save(path, backend, config):
    step = int(backend.get_optimizer_state()["step"])
    save_checkpoint(path, backend.export_weights(), backend.get_optimizer_state(), {
        "config": config, "step": step,
        "next_episode": step * config["training"]["accumulation"],
    })


@unittest.skipUnless(HAS_TORCH, "Optional PyTorch runtime is not installed")
class SameBackendResumeTests(unittest.TestCase):
    def test_disk_resume_matches_uninterrupted_next_update(self):
        config = _config()
        original = TorchBackend(config["model"], initialize_weights(config["model"], seed=42),
                                optimizer_config=config["optimizer"])
        batches = _batches(config)
        original.train_step(batches, lr=0.0003)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            _save(checkpoint, original, config)
            weights, moments, metadata = load_checkpoint(checkpoint)
            self.assertEqual(metadata["trainer"]["next_episode"], 2)
            restored = TorchBackend(config["model"], weights, optimizer_config=config["optimizer"])
            restored.set_optimizer_state(moments)
            original.train_step(list(reversed(batches)), lr=0.0002)
            restored.train_step(list(reversed(batches)), lr=0.0002)
            for name, value in original.export_weights().items():
                np.testing.assert_array_equal(value, restored.export_weights()[name])
            for name, value in original.get_optimizer_state().items():
                np.testing.assert_array_equal(value, restored.get_optimizer_state()[name])


@unittest.skipUnless(HAS_TORCH and HAS_MLX, "Both optional MLX and PyTorch runtimes are required")
class CrossBackendPortabilityTests(unittest.TestCase):
    def test_loaded_mlx_moments_and_predictions_survive_migration(self):
        from seeyon_tab.parity import run_parity
        config = _config("mlx")
        source = MLXBackend(config["model"], initialize_weights(config["model"], seed=42),
                            device="cpu", optimizer_config=config["optimizer"])
        source.train_step(_batches(config), lr=0.0003)
        self.assertTrue(any(np.any(value != 0) for key, value in source.get_optimizer_state().items()
                            if key.startswith("m::")))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "source"
            output = Path(directory) / "parity.json"
            _save(checkpoint, source, config)
            original_hashes = {p.name: sha256(p) for p in checkpoint.iterdir()}
            report = run_parity(checkpoint, mlx_device="cpu", torch_device="cpu", output=output)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["source_optimizer_step"], 1)
            self.assertTrue(report["diagnostic_update"]["optimizer_step_correct"])
            self.assertEqual(report["diagnostic_update"]["mlx"]["step"], 2)
            self.assertTrue(all(report["optimizer_state_exactly_restored"].values()))
            for phase in ("before_update", "after_update"):
                self.assertEqual(set(report[phase]), {"classification", "regression"})
                self.assertTrue(all(value["passed"] for value in report[phase].values()))
            self.assertEqual(json.loads(output.read_text()), report)
            self.assertEqual(original_hashes, {p.name: sha256(p) for p in checkpoint.iterdir()})

    def test_parity_failure_is_a_gate_and_writes_failed_report(self):
        from unittest.mock import patch
        from seeyon_tab.parity import run_parity
        original_predict = TorchBackend.predict

        def shifted_prediction(backend, batch):
            return original_predict(backend, batch) + np.float32(0.01)

        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "failed.json"
            with patch.object(TorchBackend, "predict", shifted_prediction):
                with self.assertRaises(AssertionError):
                    run_parity(output=report_path)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["before_update"]["classification"]["passed"])

    def test_default_parity_builds_and_roundtrips_nonzero_optimizer_state(self):
        from seeyon_tab.parity import run_parity
        report = run_parity()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["source_optimizer_step"], 2)
        self.assertEqual(report["diagnostic_update"]["mlx"]["step"], 3)
        self.assertEqual(report["diagnostic_update"]["torch"]["step"], 3)
        self.assertFalse(report["source_checkpoint_modified"])


if __name__ == "__main__":
    unittest.main()
