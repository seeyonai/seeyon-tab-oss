"""Write a small inspectable corpus. Usage: python -m seeyon_tab.data.generate."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .factory import GENERATOR_VERSION, SPLIT_FAMILIES, sample_episode


def create_pilot(output: Path, counts: dict[str, int], seed: int = 20261123,
                 n_support: int = 128, n_query: int = 64, n_features: int = 16) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    tasks_dir = output / "episodes"
    tasks_dir.mkdir(exist_ok=True)
    existing = list(tasks_dir.glob("*.npz"))
    if existing:
        raise FileExistsError(f"{tasks_dir} already has episodes; choose a new directory")
    rows_total = 0
    feature_values = 0
    missing_values = 0
    family_counts: Counter = Counter()
    task_counts: Counter = Counter()
    workflow_examples: list[dict] = []
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for split, count in counts.items():
            for i in range(count):
                # Balanced mechanism and task scheduling makes the tiny corpus
                # inspectable.  Large-scale training should draw online instead.
                family = SPLIT_FAMILIES[split][i % len(SPLIT_FAMILIES[split])]
                n_families = len(SPLIT_FAMILIES[split])
                problem = "classification" if (i // n_families + i % n_families) % 2 == 0 else "regression"
                episode = sample_episode(seed + i, split, family, n_support, n_query, n_features, problem)
                meta = episode["metadata"]
                relative = f"episodes/{split}_{i:04d}.npz"
                np.savez_compressed(output / relative,
                                    **{key: episode[key] for key in ("X_support", "y_support", "X_query", "y_query")})
                manifest.write(json.dumps({**meta, "array_path": relative}, ensure_ascii=False) + "\n")
                rows_total += n_support + n_query
                X = np.concatenate((episode["X_support"], episode["X_query"]))
                feature_values += X.size
                missing_values += int(np.isnan(X).sum())
                family_counts[f"{split}/{family}"] += 1
                task_counts[f"{split}/{problem}"] += 1
                if family.startswith("workflow_") and not any(e["metadata"]["split"] == split for e in workflow_examples):
                    workflow_examples.append(episode)

    preview = output / "workflow_preview.csv"
    if workflow_examples:
        names = [field["name"] for field in workflow_examples[0]["metadata"]["feature_schema"]]
        with preview.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["episode_id", "split", "role", *names, "submission_time_hours",
                             "completion_time_hours", "duration_hours", "target_name", "target_value"])
            for episode in workflow_examples:
                meta = episode["metadata"]
                for role in ("support", "query"):
                    X, y = episode[f"X_{role}"], episode[f"y_{role}"]
                    for i, row in enumerate(X):
                        submitted = meta[f"{role}_submission_time"][i]
                        completed = meta[f"{role}_completion_time"][i]
                        writer.writerow([meta["episode_id"], meta["split"], role,
                                         *["" if np.isnan(value) else float(value) for value in row],
                                         submitted, completed, completed - submitted, meta["target"], float(y[i])])
    summary = {
        "generator_version": GENERATOR_VERSION, "base_seed": seed,
        "episodes": int(sum(counts.values())), "rows": rows_total,
        "feature_count": n_features, "rows_per_episode": n_support + n_query,
        "support_per_episode": n_support, "query_per_episode": n_query,
        "episode_counts": counts, "family_counts": dict(family_counts),
        "task_counts": dict(task_counts),
        "missing_feature_fraction": missing_values / max(feature_values, 1),
        "provenance": "Generated locally by the Seeyon Tab synthetic episode factory.",
        "status": "Illustrative research pilot, not sufficient foundation-model pretraining data.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "DATA_CARD.md").write_text(
        "# Seeyon Tab synthetic pilot\n\n"
        f"**{summary['episodes']} episodes, {rows_total:,} rows, {n_features} features per row.** "
        "All values are synthetic. This corpus is a reproducible engineering pilot, not a representation "
        "of Seeyon customer records or a sufficient foundation-model pretraining run.\n\n"
        "## Read the files\n\n"
        "- `manifest.jsonl`: episode provenance, seeds, task types, schema, paths, and audit metadata.\n"
        "- `episodes/*.npz`: `X_support`, `y_support`, `X_query`, `y_query`; use `numpy.load(..., allow_pickle=False)`.\n"
        "- `workflow_preview.csv`: human-inspection sample. Its IDs, split, role, completion time and target columns "
        "are **not predictor inputs**. The declared schema alone identifies input features.\n"
        "- `summary.json`: exact counts and measured missingness.\n\n"
        "## Prediction and leakage rules\n\n"
        "1. Fit imputation, scaling, categorical mappings, and target normalization on support only.\n"
        "2. Never give `y_query` or simulator audit metadata to the model.\n"
        "3. Queries may attend support and themselves, but not other query rows: a later workflow snapshot "
        "can reveal progress that was unknown at an earlier query.\n"
        "4. For workflows, every support completion is before the earliest query submission. "
        "Unfinished historical cases can remain in the observed queue but do not provide support labels. "
        "This completed-label selection underrepresents long-running cases; survival/censoring modeling is future work.\n"
        "5. `test_ood` contains radial, piecewise, multiplicative and rework mechanisms prohibited in training/validation. "
        "Train, validation and both tests use disjoint synthetic-world namespaces. Freeze test sets before modeling; "
        "a developer can still deliberately change the code, so the guard is procedural rather than cryptographic.\n\n"
        "## Mechanisms and limits\n\n"
        "Generic worlds combine correlated feature SCMs, mixed numeric/categorical marginals, shuffled category codes, "
        "heavy-tailed heteroskedastic noise and value-dependent missingness. Workflow worlds combine department "
        "capacity queues, staged serial/parallel approval paths, policy SLAs, calendar delays, latent execution noise "
        "and mild query-time case-mix shift. The held-out workflow mechanism introduces rework loops.\n\n"
        "One workflow case occupies one department server; the staged graph controls its service requirement. "
        "This is not a full cross-department event simulator. The calendar has weekends/nights and no public holidays. "
        "Categorical codes are arbitrary within each world and vary between worlds. Missingness is not dependent on future labels. "
        "These priors are hypotheses, not calibrated enterprise statistics. Causal truth exists only inside this simulator; "
        "prediction accuracy cannot establish real-world intervention effects.\n\n"
        "The exported small corpus balances families/tasks for inspection. Use the online factory to draw fresh worlds "
        "during larger experiments; revise mixture weights only on development evidence.\n",
        encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/pilot"))
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--train", type=int, default=96)
    parser.add_argument("--validation", type=int, default=24)
    parser.add_argument("--test-iid", type=int, default=24)
    parser.add_argument("--test-ood", type=int, default=24)
    parser.add_argument("--support", type=int, default=128)
    parser.add_argument("--query", type=int, default=64)
    parser.add_argument("--features", type=int, default=16)
    args = parser.parse_args()
    counts = {"train": args.train, "validation": args.validation, "test_iid": args.test_iid, "test_ood": args.test_ood}
    if any(count < 0 for count in counts.values()):
        parser.error("Episode counts must be nonnegative")
    print(json.dumps(create_pilot(args.output, counts, args.seed, args.support, args.query, args.features), indent=2))


if __name__ == "__main__":
    main()
