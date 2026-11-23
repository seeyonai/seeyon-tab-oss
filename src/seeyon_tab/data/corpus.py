"""Validated NumPy-only episode loading for both MLX and Torch.

The manifest contains audit information. Only model_metadata() is permitted
across the model input boundary. This is a data-contract guard, not a promise
that arbitrary imported business data is free of semantic leakage.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .factory import SPLIT_FAMILIES

ARRAY_KEYS = ("X_support", "y_support", "X_query", "y_query")
MODEL_METADATA_KEYS = ("problem_type", "categorical_indices", "feature_schema")
_FORBIDDEN_FEATURE_NAMES = {
    "duration_hours", "sla_breach", "completion_time", "completion_time_hours",
    "rework_loops", "episode_id", "enterprise_or_world_id", "target_value",
}


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_metadata(meta: dict[str, Any]) -> None:
    if not isinstance(meta, dict):
        raise ValueError("Episode metadata must be an object")
    split = meta.get("split")
    if split not in SPLIT_FAMILIES:
        raise ValueError(f"Unknown split: {split!r}")
    if meta.get("mechanism_family") not in SPLIT_FAMILIES[split]:
        raise ValueError("Mechanism is prohibited in its declared split")
    if meta.get("problem_type") not in ("classification", "regression"):
        raise ValueError("Only classification/regression tasks are supported")
    for key in ("episode_id", "enterprise_or_world_id", "generator_version"):
        if not isinstance(meta.get(key), str) or not meta[key]:
            raise ValueError(f"Missing identity field: {key}")
    _positive_int(meta.get("n_support"), "n_support")
    _positive_int(meta.get("n_query"), "n_query")
    d = _positive_int(meta.get("n_features"), "n_features")
    schema = meta.get("feature_schema")
    if not isinstance(schema, list) or len(schema) != d:
        raise ValueError("Feature schema length differs from n_features")
    names = []
    expected_categories = []
    for j, field in enumerate(schema):
        if not isinstance(field, dict) or not isinstance(field.get("name"), str):
            raise ValueError("Every feature needs a declared name and type")
        name = field["name"]
        if not name or name in _FORBIDDEN_FEATURE_NAMES:
            raise ValueError(f"Forbidden or empty predictive feature: {name!r}")
        names.append(name)
        if field.get("type") not in ("numeric", "categorical"):
            raise ValueError("Feature type must be numeric or categorical")
        if field.get("available_at") not in ("prediction_time", "submission_time"):
            raise ValueError("Every feature needs an explicit prediction-time contract")
        if field["type"] == "categorical":
            expected_categories.append(j)
    if len(set(names)) != d:
        raise ValueError("Duplicate feature names")
    if meta.get("categorical_indices") != expected_categories:
        raise ValueError("Categorical indices disagree with schema")
    if meta["mechanism_family"].startswith("workflow_"):
        if meta.get("query_isolation_required") is not True:
            raise ValueError("Workflow episodes require isolated query predictions")
        for role, n in (("support", meta["n_support"]), ("query", meta["n_query"])):
            submitted = np.asarray(meta.get(f"{role}_submission_time", []), dtype=float)
            completed = np.asarray(meta.get(f"{role}_completion_time", []), dtype=float)
            if submitted.shape != (n,) or completed.shape != (n,):
                raise ValueError("Workflow audit timestamp shape mismatch")
            if not np.isfinite(submitted).all() or not np.isfinite(completed).all():
                raise ValueError("Non-finite audit timestamp")
            if not np.all(completed > submitted):
                raise ValueError("Completion must occur after submission")
        if max(meta["support_completion_time"]) > min(meta["query_submission_time"]):
            raise ValueError("A support label is unavailable at query time")


def validate_episode(episode: dict[str, Any]) -> None:
    """Validate structure, task labels and temporal visibility without fitting."""
    if set(episode) != {*ARRAY_KEYS, "metadata"}:
        raise ValueError("An episode must have exactly four arrays and metadata")
    meta = episode["metadata"]
    validate_metadata(meta)
    for role in ("support", "query"):
        n = meta[f"n_{role}"]
        x = np.asarray(episode[f"X_{role}"])
        y = np.asarray(episode[f"y_{role}"])
        if x.shape != (n, meta["n_features"]) or y.shape != (n,):
            raise ValueError("Episode array shape disagrees with manifest")
        if x.dtype.kind not in "fiu" or y.dtype.kind not in "fiu":
            raise ValueError("Only numeric arrays are allowed; pickle/object arrays are forbidden")
        if np.isinf(x).any() or not np.isfinite(y).all():
            raise ValueError("Features may be NaN, but infinities/missing labels are forbidden")
        if meta["problem_type"] == "classification" and not np.isin(y, [0, 1]).all():
            raise ValueError("Classification labels must be binary 0/1")


def model_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Return only the explicit schema/task allowlist, never simulator truth."""
    return {key: copy.deepcopy(meta[key]) for key in MODEL_METADATA_KEYS}


def prediction_view(episode: dict[str, Any]) -> dict[str, Any]:
    """Omit query labels and audit truth at a prediction-facing API boundary."""
    return {
        "X_support": episode["X_support"],
        "y_support": episode["y_support"],
        "X_query": episode["X_query"],
        "metadata": model_metadata(episode["metadata"]),
    }


class EpisodeCorpus(Sequence):
    """A validated disk split. Training access requires for_training=True.

    Only train may be opened with for_training=True. Merely selecting an eval
    split does not make it hidden: the shipped pilot is a transparent diagnostic.
    Arrays load lazily and are validated every time, with allow_pickle=False.
    """

    def __init__(self, directory: str | Path, split: str, *, for_training: bool = False):
        if split not in SPLIT_FAMILIES:
            raise ValueError(f"Unknown split: {split!r}")
        if for_training and split != "train":
            raise ValueError("Training may only use the train split")
        self.directory = Path(directory).resolve()
        self.split = split
        self.for_training = bool(for_training)
        self.records: list[dict[str, Any]] = []
        all_ids = set()
        world_splits: dict[str, str] = {}
        manifest = self.directory / "manifest.jsonl"
        with manifest.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    meta = json.loads(line)
                    validate_metadata(meta)
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError(f"Invalid manifest line {lineno}: {exc}") from exc
                if meta["episode_id"] in all_ids:
                    raise ValueError("Duplicate episode ID in corpus")
                all_ids.add(meta["episode_id"])
                world = meta["enterprise_or_world_id"]
                if world in world_splits and world_splits[world] != meta["split"]:
                    raise ValueError("A synthetic world appears in multiple splits")
                world_splits[world] = meta["split"]
                self._array_path(meta)
                if meta["split"] == split:
                    self.records.append(meta)
        if not self.records:
            raise ValueError(f"No episodes in split {split!r}")

    def _array_path(self, record: dict[str, Any]) -> Path:
        relative = record.get("array_path")
        if not isinstance(relative, str) or not relative:
            raise ValueError("Missing relative array_path")
        raw = Path(relative)
        if raw.is_absolute():
            raise ValueError("Absolute array paths are forbidden")
        resolved = (self.directory / raw).resolve()
        if not resolved.is_relative_to(self.directory) or resolved.suffix != ".npz":
            raise ValueError("Array path must stay inside the corpus and end in .npz")
        if not resolved.is_file():
            raise ValueError(f"Episode file missing: {relative}")
        return resolved

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, slice):
            raise TypeError("Load episodes by integer index to preserve lazy validation")
        record = self.records[index]
        path = self._array_path(record)
        if "array_sha256" in record:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != record["array_sha256"]:
                raise ValueError("Episode checksum differs from manifest")
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != set(ARRAY_KEYS):
                raise ValueError("Unexpected or missing NPZ fields")
            episode = {key: arrays[key].copy() for key in ARRAY_KEYS}
        episode["metadata"] = copy.deepcopy(record)
        validate_episode(episode)
        return episode

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]

    def audit(self) -> dict[str, Any]:
        """Load every selected episode and return descriptive corpus statistics."""
        family_counts: Counter = Counter()
        task_counts: Counter = Counter()
        total_rows = 0
        feature_values = 0
        missing_values = 0
        workflow_episodes = 0
        for episode in self:
            meta = episode["metadata"]
            family_counts[meta["mechanism_family"]] += 1
            task_counts[meta["problem_type"]] += 1
            total_rows += meta["n_support"] + meta["n_query"]
            workflow_episodes += int(meta["mechanism_family"].startswith("workflow_"))
            for role in ("support", "query"):
                x = episode[f"X_{role}"]
                feature_values += x.size
                missing_values += int(np.isnan(x).sum())
        return {
            "split": self.split, "episodes": len(self), "rows": total_rows,
            "workflow_episodes_with_visibility_check": workflow_episodes,
            "family_counts": dict(family_counts), "task_counts": dict(task_counts),
            "missing_feature_fraction": missing_values / feature_values,
            "all_episode_structural_and_visibility_checks_passed": True,
        }

