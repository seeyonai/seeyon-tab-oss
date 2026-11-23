"""MLX behavior checks; skipped only when the optional runtime is unavailable."""
from __future__ import annotations

import copy
import os
import unittest

import numpy as np

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except (ImportError, OSError):
    MLX_AVAILABLE = False

from seeyon_tab.weights import initialize_weights
from seeyon_tab.backends.mlx_backend import MLXBackend


MODEL = {
    "width": 16, "heads": 4, "row_layers": 1, "icl_layers": 2,
    "ff_multiplier": 2, "max_features": 16, "category_vocab": 66,
}
TEST_DEVICE = os.environ.get("SEEYON_TEST_MLX_DEVICE", "cpu")


def batch(task="classification"):
    rng = np.random.default_rng(171)
    numeric = rng.normal(size=(12, 4)).astype(np.float32)
    numeric[:, 2] = 0
    categories = np.zeros((12, 4), dtype=np.int32)
    categories[:, 2] = np.arange(12) % 3 + 2
    missing = np.zeros((12, 4), dtype=bool)
    missing[3, 1] = True
    numeric[3, 1] = 0
    labels = np.arange(8) % 2 if task == "classification" else np.sin(np.arange(8))
    targets = np.arange(4) % 2 if task == "classification" else np.cos(np.arange(4))
    return {
        "numeric": numeric, "categories": categories, "missing": missing,
        "is_categorical": np.array([False, False, True, False]),
        "y_support": labels.astype(np.float32),
        "targets": targets.astype(np.float32), "task": task,
    }


@unittest.skipUnless(MLX_AVAILABLE, "Optional MLX runtime is not installed")
class MLXBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.weights = initialize_weights(MODEL, seed=942)
        self.backend = MLXBackend(MODEL, self.weights, device=TEST_DEVICE)

    def test_both_heads_are_finite_and_quantiles_ordered(self):
        logits = self.backend.predict(batch("classification"))
        self.assertEqual(logits.shape, (4, 2))
        self.assertTrue(np.isfinite(logits).all())
        quantiles = self.backend.predict(batch("regression"))
        self.assertEqual(quantiles.shape, (4, 3))
        self.assertTrue(np.isfinite(quantiles).all())
        self.assertTrue(np.all(np.diff(quantiles, axis=-1) >= 0))

    def test_no_query_to_query_leakage_or_batch_dependence(self):
        original = batch()
        predicted = self.backend.predict(original)[0]
        changed = copy.deepcopy(original)
        changed["numeric"][9:] *= -13
        changed["categories"][9:, 2] = 5
        np.testing.assert_allclose(
            predicted, self.backend.predict(changed)[0], rtol=2e-5, atol=2e-5
        )
        single = copy.deepcopy(original)
        for key in ("numeric", "categories", "missing"):
            single[key] = single[key][:9]
        single["targets"] = single["targets"][:1]
        np.testing.assert_allclose(
            predicted, self.backend.predict(single)[0], rtol=2e-5, atol=2e-5
        )

    def test_support_row_permutation_invariance(self):
        original = batch("regression")
        expected = self.backend.predict(original)
        permuted = copy.deepcopy(original)
        order = np.array([6, 0, 4, 7, 1, 5, 2, 3])
        row_order = np.r_[order, np.arange(8, 12)]
        for key in ("numeric", "categories", "missing"):
            permuted[key] = permuted[key][row_order]
        permuted["y_support"] = permuted["y_support"][order]
        np.testing.assert_allclose(
            expected, self.backend.predict(permuted), rtol=3e-5, atol=3e-5
        )

    def test_query_targets_are_ignored_and_unlabeled_inference_works(self):
        original = batch()
        expected = self.backend.predict(original)
        changed = copy.deepcopy(original)
        changed["targets"] = np.array([1000, -2000, np.nan, np.inf])
        np.testing.assert_array_equal(expected, self.backend.predict(changed))
        del changed["targets"]
        np.testing.assert_array_equal(expected, self.backend.predict(changed))

    def test_gradient_update_and_accumulation_change_parameters(self):
        report = self.backend.train_step([batch(), batch("regression")], lr=2e-4)
        self.assertTrue(np.isfinite(report["loss"]))
        self.assertTrue(np.isfinite(report["grad_norm"]))
        self.assertGreater(report["grad_norm"], 0)
        self.assertEqual(int(self.backend.get_optimizer_state()["step"]), 1)
        after = self.backend.export_weights()
        self.assertTrue(any(not np.array_equal(after[key], self.weights[key]) for key in after))
        self.assertTrue(all(np.isfinite(value).all() for value in after.values()))
        with self.assertRaises(ValueError):
            self.backend.train_step([], lr=2e-4)

    def test_optimizer_state_resume_matches_uninterrupted_update(self):
        self.backend.train_step([batch(), batch("regression")], lr=2e-4)
        restored = MLXBackend(MODEL, self.backend.export_weights(), device=TEST_DEVICE)
        restored.set_optimizer_state(self.backend.get_optimizer_state())
        self.backend.train_step([batch("regression")], lr=1e-4)
        restored.train_step([batch("regression")], lr=1e-4)
        first, second = self.backend.export_weights(), restored.export_weights()
        for key in first:
            np.testing.assert_allclose(first[key], second[key], rtol=1e-6, atol=1e-6)
        self.assertEqual(int(restored.get_optimizer_state()["step"]), 2)

    def test_invalid_optimizer_restore_is_rejected(self):
        state = self.backend.get_optimizer_state()
        state["step"] = np.asarray(-1, dtype=np.int64)
        with self.assertRaises(ValueError):
            self.backend.set_optimizer_state(state)

    def test_partial_config_and_large_category_ids(self):
        partial = {"width": 16, "heads": 4}
        model = MLXBackend(partial, initialize_weights(partial, seed=7), device=TEST_DEVICE)
        self.assertEqual(model.predict(batch()).shape, (4, 2))
        invalid = batch()
        invalid["categories"] = invalid["categories"].astype(np.int64)
        invalid["categories"][0, 2] = 2 ** 32 + 2
        with self.assertRaises(ValueError):
            self.backend.predict(invalid)


if __name__ == "__main__":
    unittest.main()

