"""Behavioral checks for portable Torch training and strict ICL visibility."""
from __future__ import annotations

import copy
import importlib.util
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from seeyon_tab.backends.torch_backend import TorchBackend
    from seeyon_tab.weights import initialize_weights
    torch.set_num_threads(2)


@unittest.skipUnless(HAS_TORCH, "Optional PyTorch runtime is not installed")
class TorchBackendTests(unittest.TestCase):
    def setUp(self):
        self.config = {"width": 16, "heads": 4, "row_layers": 1, "icl_layers": 2,
                       "ff_multiplier": 2, "max_features": 8, "category_vocab": 10}
        rng = np.random.default_rng(37)
        self.weights = initialize_weights(self.config, seed=37)
        self.backend = TorchBackend(self.config, self.weights)
        self.batch = {
            "numeric": rng.normal(size=(9, 5)).astype(np.float32),
            "categories": rng.integers(0, 6, size=(9, 5), dtype=np.int32),
            "missing": np.zeros((9, 5), dtype=bool),
            "is_categorical": np.array([False, True, False, False, True]),
            "y_support": np.array([0, 1, 1, 0, 1, 0], np.float32),
            "task": "classification",
            "targets": np.array([1, 0, 1], np.float32),
        }

    def test_query_labels_are_ignored_for_prediction(self):
        original = self.backend.predict(self.batch)
        changed = copy.deepcopy(self.batch)
        changed["targets"] = np.full(3, np.nan, dtype=np.float32)
        np.testing.assert_array_equal(original, self.backend.predict(changed))
        changed.pop("targets")
        np.testing.assert_array_equal(original, self.backend.predict(changed))
        with self.assertRaises(ValueError):
            self.backend.train_step([changed], lr=0.001)

    def test_queries_cannot_influence_each_other(self):
        original = self.backend.predict(self.batch)
        changed = copy.deepcopy(self.batch)
        changed["numeric"][7:] = 999.0
        changed["categories"][7:] = 8
        changed["missing"][7:] = True
        np.testing.assert_allclose(original[0], self.backend.predict(changed)[0], atol=2e-6, rtol=2e-6)
        single = {key: (value[:7].copy() if key in ("numeric", "categories", "missing") else value)
                  for key, value in self.batch.items()}
        single["targets"] = single["targets"][:1]
        np.testing.assert_allclose(original[:1], self.backend.predict(single), atol=2e-6, rtol=2e-6)

    def test_support_permutation_invariance(self):
        original = self.backend.predict(self.batch)
        changed = copy.deepcopy(self.batch)
        order = np.array([4, 2, 0, 5, 1, 3])
        for key in ("numeric", "categories", "missing"):
            changed[key][:6] = changed[key][order]
        changed["y_support"] = changed["y_support"][order]
        np.testing.assert_allclose(original, self.backend.predict(changed), atol=2e-6, rtol=2e-6)

    def test_actual_training_and_ordered_quantiles(self):
        before = self.backend.export_weights()
        metrics = self.backend.train_step([self.batch], lr=0.001)
        self.assertTrue(np.isfinite(metrics["loss"]))
        self.assertGreater(metrics["grad_norm"], 0)
        self.assertTrue(any(not np.array_equal(before[k], v) for k, v in self.backend.export_weights().items()))
        regression = copy.deepcopy(self.batch)
        regression["task"] = "regression"
        regression["y_support"] = np.array([-1.1, 0.2, 1.5, 0.7, -0.3, 0.1], np.float32)
        regression["targets"] = np.array([0.1, 2, -0.5], np.float32)
        regression_metrics = self.backend.train_step([regression], lr=0.001)
        self.assertTrue(np.isfinite(regression_metrics["loss"]))
        self.assertTrue(np.all(np.diff(self.backend.predict(regression), axis=-1) >= 0))

    def test_resume_preserves_next_update(self):
        self.backend.train_step([self.batch], lr=0.001)
        resumed = TorchBackend(self.config, self.backend.export_weights())
        resumed.set_optimizer_state(self.backend.get_optimizer_state())
        self.backend.train_step([self.batch], lr=0.0007)
        resumed.train_step([self.batch], lr=0.0007)
        self.assertEqual(int(resumed.get_optimizer_state()["step"]), 2)
        for key, value in self.backend.export_weights().items():
            np.testing.assert_array_equal(value, resumed.export_weights()[key])
        for key, value in self.backend.get_optimizer_state().items():
            np.testing.assert_array_equal(value, resumed.get_optimizer_state()[key])

    def test_accumulation_averages_episode_gradients(self):
        repeated = TorchBackend(self.config, self.weights)
        one = self.backend.train_step([self.batch], lr=0.001)
        two = repeated.train_step([self.batch, self.batch], lr=0.001)
        self.assertAlmostEqual(one["loss"], two["loss"], places=6)
        for key, value in self.backend.export_weights().items():
            np.testing.assert_allclose(value, repeated.export_weights()[key], atol=2e-6, rtol=2e-6)

    def test_corrupt_weights_or_optimizer_state_rejected(self):
        broken_weights = dict(self.weights)
        broken_weights.pop("row_token")
        with self.assertRaises(ValueError):
            TorchBackend(self.config, broken_weights)
        state = self.backend.get_optimizer_state()
        state["v::row_token"][0] = -1.0
        with self.assertRaises(ValueError):
            self.backend.set_optimizer_state(state)
        self.assertEqual(int(self.backend.get_optimizer_state()["step"]), 0)


if __name__ == "__main__":
    unittest.main()
