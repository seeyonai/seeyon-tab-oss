"""Data contracts that protect reproducibility and the prediction boundary."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from seeyon_tab.data import (
    EpisodeCorpus, SPLIT_FAMILIES, sample_episode,
    model_metadata, prediction_view, validate_episode,
)
from seeyon_tab.data.generate import create_pilot


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.pilot = Path(directory.name)
        counts = {"train": 96, "validation": 24, "test_iid": 24, "test_ood": 24}
        create_pilot(cls.pilot, counts)

    def test_resolved_configuration_replays_identically(self):
        episode = sample_episode(51, n_support=24, n_query=12)
        meta = episode["metadata"]
        explicit = sample_episode(
            51, family=meta["mechanism_family"], problem_type=meta["problem_type"],
            n_support=24, n_query=12,
        )
        for key in ("X_support", "X_query", "y_support", "y_query"):
            np.testing.assert_array_equal(episode[key], explicit[key])
        self.assertEqual(meta, explicit["metadata"])
        changed = sample_episode(51, family=meta["mechanism_family"],
                                 problem_type=meta["problem_type"], n_support=25, n_query=12)
        self.assertNotEqual(meta["episode_id"], changed["metadata"]["episode_id"])

    def test_held_out_mechanisms_and_worlds(self):
        self.assertTrue(set(SPLIT_FAMILIES["train"]).isdisjoint(SPLIT_FAMILIES["test_ood"]))
        with self.assertRaises(ValueError):
            sample_episode(1, split="train", family="workflow_rework")
        worlds = [
            sample_episode(100, split=split, family="linear", n_support=8, n_query=4)
            ["metadata"]["enterprise_or_world_id"]
            for split in ("train", "validation", "test_iid")
        ]
        self.assertEqual(len(set(worlds)), 3)

    def test_all_workflow_support_labels_are_known(self):
        for split, families in SPLIT_FAMILIES.items():
            for family in families:
                if family.startswith("workflow_"):
                    episode = sample_episode(19, split, family, n_support=32, n_query=12)
                    validate_episode(episode)
                    meta = episode["metadata"]
                    self.assertLessEqual(max(meta["support_completion_time"]),
                                         min(meta["query_submission_time"]))

    def test_paired_tasks_share_observations(self):
        for split, family in (("train", "linear"), ("train", "workflow_parallel")):
            cls = sample_episode(11, split, family, 24, 8, problem_type="classification")
            reg = sample_episode(11, split, family, 24, 8, problem_type="regression")
            np.testing.assert_array_equal(cls["X_support"], reg["X_support"])
            np.testing.assert_array_equal(cls["X_query"], reg["X_query"])
            self.assertNotEqual(cls["metadata"]["episode_id"], reg["metadata"]["episode_id"])

    def test_generated_corpus_and_training_split_guard(self):
        corpus = EpisodeCorpus(self.pilot, "train", for_training=True)
        self.assertEqual(len(corpus), 96)
        self.assertEqual(corpus[0]["X_support"].shape, (128, 16))
        with self.assertRaisesRegex(ValueError, "Training"):
            EpisodeCorpus(self.pilot, "test_ood", for_training=True)
        total = sum(EpisodeCorpus(self.pilot, split).audit()["rows"] for split in SPLIT_FAMILIES)
        self.assertEqual(total, 32256)

    def test_model_boundary_has_no_query_labels_or_simulator_truth(self):
        episode = sample_episode(17, family="workflow_parallel", n_support=12, n_query=4)
        view = prediction_view(episode)
        self.assertNotIn("y_query", view)
        self.assertEqual(set(view["metadata"]), {"problem_type", "categorical_indices", "feature_schema"})
        for key in ("query_completion_time", "episode_id", "split", "enterprise_or_world_id"):
            self.assertNotIn(key, view["metadata"])
        view["metadata"]["feature_schema"][0]["name"] = "changed"
        self.assertNotEqual(episode["metadata"]["feature_schema"][0]["name"], "changed")

    def test_shape_and_timestamp_corruption_are_rejected(self):
        episode = sample_episode(22, family="workflow_parallel", n_support=12, n_query=4)
        broken = copy.deepcopy(episode)
        broken["X_query"] = broken["X_query"][:-1]
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_episode(broken)
        broken = copy.deepcopy(episode)
        broken["metadata"]["support_completion_time"][0] = (
            max(broken["metadata"]["query_completion_time"]) + 100
        )
        with self.assertRaisesRegex(ValueError, "unavailable"):
            validate_episode(broken)

    def test_manifest_cannot_escape_directory(self):
        record = json.loads((self.pilot / "manifest.jsonl").read_text().splitlines()[0])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record["array_path"] = "../elsewhere.npz"
            (root / "manifest.jsonl").write_text(json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "inside"):
                EpisodeCorpus(root, "train", for_training=True)

    def test_npz_object_arrays_are_rejected(self):
        record = json.loads((self.pilot / "manifest.jsonl").read_text().splitlines()[0])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record["array_path"] = "object.npz"
            (root / "manifest.jsonl").write_text(json.dumps(record) + "\n")
            np.savez(root / "object.npz", X_support=np.array([object()], dtype=object),
                     X_query=np.zeros((64, 16)), y_support=np.zeros(128), y_query=np.zeros(64))
            corpus = EpisodeCorpus(root, "train")
            with self.assertRaises(ValueError):
                corpus[0]


if __name__ == "__main__":
    unittest.main()

