"""A small, reproducible *research pilot*, not a calibrated enterprise digital twin.

The generator uses synthetic data. Each task is an
independent synthetic world.  The final OOD mechanisms are never sampled by the
train/validation/IID APIs.  All predictive arrays exclude task IDs and simulator
state.  Workflow support labels are completed before the first query arrives.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

GENERATOR_VERSION = "seeyon-tab-pilot-0.1.0"
SPLIT_FAMILIES = {
    "train": ("linear", "additive", "interactions", "workflow_serial", "workflow_parallel"),
    "validation": ("linear", "additive", "interactions", "workflow_serial", "workflow_parallel"),
    "test_iid": ("linear", "additive", "interactions", "workflow_serial", "workflow_parallel"),
    "test_ood": ("radial", "piecewise", "multiplicative", "workflow_rework"),
}
_SPLIT_NAMESPACE = {"train": 1101, "validation": 2202, "test_iid": 3303, "test_ood": 4404}

WORKFLOW_FEATURES = [
    ("log_amount", "numeric"),
    ("budget_utilization", "numeric"),
    ("urgency", "categorical"),
    ("planned_stages", "numeric"),
    ("planned_parallel_width", "numeric"),
    ("department_capacity", "numeric"),
    ("queue_length_at_submission", "numeric"),
    ("department_recent_cycle_hours", "numeric"),
    ("log_organization_size", "numeric"),
    ("submission_weekday", "categorical"),
    ("submission_hour", "numeric"),
    ("case_type", "categorical"),
    ("department", "categorical"),
    ("planned_external_dependency", "categorical"),
    ("sla_hours", "numeric"),
    ("submission_business_day", "numeric"),
]


def _business_to_wall(t: float | np.ndarray) -> float | np.ndarray:
    """Business hour zero = Monday 09:00; 8 h/day and 5 days/week.

    Returns elapsed calendar hours from that origin.  This deliberately simple
    calendar has weekends and nights but no country-specific holidays.
    """
    a = np.asarray(t)
    weeks = np.floor(a / 40.0)
    within_week = a - 40.0 * weeks
    days = np.floor(within_week / 8.0)
    hours = within_week - 8.0 * days
    out = weeks * 168.0 + days * 24.0 + hours
    return float(out) if out.ndim == 0 else out


def _apply_missingness(X: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Include value-dependent missingness; never condition missingness on y."""
    rate = float(rng.uniform(0.0, 0.12))
    p = np.full(X.shape, rate)
    # The first field becomes more often absent at high values: a simple MNAR
    # mechanism.  Threshold comes from the generator, not query target values.
    p[:, 0] += 0.13 / (1.0 + np.exp(-np.clip(X[:, 0] - 1.0, -20, 20)))
    out = X.astype(np.float32, copy=True)
    out[rng.random(X.shape) < p] = np.nan
    return out, rate


def _generic_world(
    rng: np.random.Generator, family: str, n: int, d: int, problem_type: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Correlated mixed-type feature SCM plus one of six target mechanisms."""
    Z = np.zeros((n, d), dtype=np.float64)
    parents: list[list[int]] = []
    for j in range(d):
        k = min(j, int(rng.integers(0, 4)))
        pa = sorted(rng.choice(j, size=k, replace=False).tolist()) if k else []
        parents.append(pa)
        noise = rng.standard_t(float(rng.uniform(3.0, 12.0)), size=n)
        Z[:, j] = 0.65 * noise
        if pa:
            Z[:, j] += np.tanh(Z[:, pa]) @ rng.normal(0, 0.6, size=k)

    X = Z.copy()
    signal = Z.copy()
    categorical: list[int] = []
    schema: list[dict[str, str]] = []
    for j in range(d):
        if rng.random() < 0.25:
            categorical.append(j)
            n_categories = int(rng.integers(2, 6))
            category = np.digitize(Z[:, j], np.linspace(-1, 1, n_categories - 1))
            permutation = rng.permutation(n_categories)
            X[:, j] = permutation[category]
            signal[:, j] = rng.normal(size=n_categories)[category]
            kind = "categorical"
        else:
            # Units and positive/skewed marginals vary across worlds.
            if rng.random() < 0.25:
                X[:, j] = np.exp(np.clip(0.5 * Z[:, j], -5, 5))
            X[:, j] = X[:, j] * 10 ** rng.uniform(-1, 2) + rng.normal(0, 3)
            kind = "numeric"
        schema.append({"name": f"feature_{j:02d}", "type": kind, "available_at": "prediction_time"})

    w = rng.normal(size=d) * rng.binomial(1, 0.65, size=d)
    if not np.any(w):
        w[0] = 1.0
    scale = max(float(np.linalg.norm(w)), 1.0)
    if family == "linear":
        f = signal @ w / scale
    elif family == "additive":
        frequency = rng.uniform(0.3, 2.5, size=d)
        f = (np.sin(signal * frequency) + 0.2 * np.tanh(signal) ** 2) @ w / scale
    elif family == "interactions":
        f = 0.4 * signal @ w / scale
        for _ in range(max(2, d // 3)):
            i, j = rng.choice(d, size=2, replace=False)
            f += rng.normal(0, 0.4) * np.tanh(signal[:, i]) * signal[:, j]
    elif family == "radial":
        # OOD: local radial effects, never used for training.
        centers = rng.normal(size=(6, d))
        widths = rng.uniform(0.4, 2.0, size=6)
        radial = np.exp(-np.mean((signal[:, None, :] - centers[None, :, :]) ** 2, axis=-1) / widths)
        f = 3.0 * (radial - 0.35) @ rng.normal(size=6)
    elif family == "piecewise":
        # OOD: discontinuous threshold/branch rules.
        f = np.zeros(n)
        for _ in range(8):
            i = int(rng.integers(d))
            threshold = float(rng.normal(0, 0.8))
            f += float(rng.normal(0, 0.6)) * (2 * (signal[:, i] > threshold) - 1)
    elif family == "multiplicative":
        # OOD: products of several effects with bounded tails.
        subset = rng.choice(d, size=min(d, 5), replace=False)
        f = 2.5 * (np.prod(1.0 + 0.5 * np.tanh(signal[:, subset]), axis=1) - 1.0)
    else:
        raise ValueError(f"Unknown generic family: {family}")

    noise_scale = float(rng.uniform(0.1, 0.8))
    heteroskedastic = noise_scale * (0.5 + np.abs(np.tanh(signal[:, 0])))
    latent = f + rng.normal() * 0.25 + heteroskedastic * rng.standard_t(6.0, size=n)
    X, missing_rate = _apply_missingness(X, rng)
    if problem_type == "classification":
        probability = 1.0 / (1.0 + np.exp(-np.clip(1.5 * latent, -20, 20)))
        y = (rng.random(n) < probability).astype(np.int64)
    else:
        y = latent.astype(np.float32)
    return X, y, {
        "feature_schema": schema,
        "categorical_indices": categorical,
        "feature_scm_parents": parents,
        "noise_scale": noise_scale,
        "nominal_missing_rate": missing_rate,
        "known_simulator_causality_only": True,
    }


def _workflow_world(
    rng: np.random.Generator, family: str, n_support: int, n_query: int,
    d: int, problem_type: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """A department queue with staged approval paths and a business calendar.

    Each department has parallel case-processing servers.  A case's service
    requirement is the critical-path duration of a staged approval graph; an
    entire case occupies one department server.  This is an explicit simplifying
    assumption: it is not a full multi-department resource-constrained simulator.
    """
    n_departments = int(rng.integers(3, 7))
    capacities = rng.integers(1, 5, size=n_departments)
    available = [np.zeros(int(cap)) for cap in capacities]
    jobs: list[dict[str, Any]] = []
    dept_speed = rng.lognormal(0.0, 0.3, size=n_departments)
    organization_size = int(rng.integers(100, 6000))
    amount_scale = float(rng.uniform(7.0, 12.0))
    policy_scale = float(rng.uniform(0.7, 1.6))
    arrival_scale = float(rng.uniform(0.6, 2.2))
    case_code = rng.permutation(4)
    dept_code = rng.permutation(n_departments)
    base_case_probability = rng.dirichlet(np.full(4, 2.0))
    shifted_case_probability = 0.75 * base_case_probability + 0.25 * rng.dirichlet(np.ones(4))
    calendar_offset_business = float(rng.integers(0, 120) * 8)

    def make_case(t: float, query: bool) -> dict[str, Any]:
        department = int(rng.integers(n_departments))
        case_type = int(rng.choice(4, p=shifted_case_probability if query else base_case_probability))
        log_amount = float(rng.normal(amount_scale + 0.3 * case_type, 1.0))
        budget = float(rng.beta(2, 2))
        urgency = int(rng.random() < (0.2 + 0.15 * (case_type == 3)))
        stages = int(np.clip(1 + case_type + (log_amount > amount_scale + 1.0) + rng.integers(0, 2), 1, 6))
        width = 1 if family == "workflow_serial" else int(rng.integers(1, 4))
        external = int(rng.random() < 0.2)
        own_jobs = [job for job in jobs if job["department"] == department]
        queue_length = sum(job["completion_business"] > t for job in own_jobs)
        completed = [job["duration_hours"] for job in own_jobs if job["completion_business"] <= t]
        recent_cycle = float(np.median(completed[-32:])) if completed else 24.0
        wall = _business_to_wall(t + calendar_offset_business)
        weekday = int((wall // 24) % 7)
        hour = 9.0 + float(wall % 24)
        sla = float([24.0, 48.0, 72.0, 120.0][case_type] * policy_scale)
        observed = [log_amount, budget, urgency, stages, width, int(capacities[department]),
                    queue_length, recent_cycle, np.log(organization_size), weekday, hour,
                    int(case_code[case_type]), int(dept_code[department]), external, sla,
                    (t + calendar_offset_business) / 8.0]

        # Latent reviewer effort and execution noise are sampled *after* the
        # as-of snapshot.  The planned width is observable; actual effort is not.
        stage_effort = rng.lognormal(np.log(1.25), 0.55, size=(stages, width))
        service = float(stage_effort.max(axis=1).sum())
        service *= float(dept_speed[department]) * (1.0 + 0.15 * (log_amount - amount_scale))
        service *= (0.75 if urgency else 1.0) * (1.0 + 0.4 * external)
        service = max(0.3, service)
        rework_loops = 0
        if family == "workflow_rework":
            # Held-out mechanism: a hidden review outcome triggers extra work;
            # it is not supplied as a feature.  Do not interpret this as evidence
            # of real-world causal effects of staffing or approval policies.
            p_rework = float(np.clip(0.15 + 0.25 * budget + 0.03 * queue_length, 0.1, 0.7))
            rework_loops = min(int(rng.geometric(1.0 - p_rework)) - 1, 4)
            service *= 1.0 + 0.65 * rework_loops
        service *= float(rng.lognormal(0, 0.25))
        server = int(np.argmin(available[department]))
        start = max(t, float(available[department][server]))
        end = start + service
        available[department][server] = end
        duration = float(_business_to_wall(end + calendar_offset_business) - wall)
        record = {
            "features": observed, "department": department,
            "submission_business": t, "completion_business": end,
            "submission_time": wall,
            "completion_time": float(_business_to_wall(end + calendar_offset_business)),
            "duration_hours": duration, "sla_breach": int(duration > sla),
        }
        jobs.append(record)
        return record

    n_history = max(3 * n_support, 48)
    arrivals = np.cumsum(rng.exponential(arrival_scale, size=n_history))
    history = [make_case(float(t), False) for t in arrivals]
    cutoff = float(arrivals[-1] + rng.uniform(4.0, 16.0))
    eligible = [job for job in history if job["completion_business"] <= cutoff]
    if len(eligible) < n_support:
        # Move the prediction origin forward until enough labels exist.  The
        # selected support still excludes any uncompleted historical case.
        cutoff = float(sorted(job["completion_business"] for job in history)[n_support - 1] + 0.001)
        eligible = [job for job in history if job["completion_business"] <= cutoff]
    support = eligible[-n_support:]
    query_arrivals = cutoff + np.cumsum(rng.exponential(arrival_scale, size=n_query))
    queries = [make_case(float(t), True) for t in query_arrivals]
    selected = support + queries
    base_X = np.asarray([job["features"] for job in selected], dtype=np.float32)
    schema = [{"name": name, "type": kind, "available_at": "submission_time"}
              for name, kind in WORKFLOW_FEATURES]
    if d <= base_X.shape[1]:
        base_X, schema = base_X[:, :d], schema[:d]
    else:
        extra = rng.normal(size=(len(selected), d - base_X.shape[1]))
        schema += [{"name": f"nuisance_{j:02d}", "type": "numeric", "available_at": "submission_time"}
                   for j in range(extra.shape[1])]
        base_X = np.column_stack((base_X, extra))
    X, missing_rate = _apply_missingness(base_X, rng)
    key = "sla_breach" if problem_type == "classification" else "duration_hours"
    y = np.asarray([job[key] for job in selected], dtype=np.int64 if problem_type == "classification" else np.float32)
    meta = {
        "feature_schema": schema,
        "categorical_indices": [j for j, field in enumerate(schema) if field["type"] == "categorical"],
        "prediction_event": "case_submission",
        "target": key,
        "duration_units": "elapsed_calendar_hours",
        "nominal_missing_rate": missing_rate,
        "support_submission_time": [job["submission_time"] for job in support],
        "support_completion_time": [job["completion_time"] for job in support],
        "query_submission_time": [job["submission_time"] for job in queries],
        "query_completion_time": [job["completion_time"] for job in queries],
        "history_candidates": n_history,
        "history_unresolved_at_cutoff": sum(job["completion_business"] > cutoff for job in history),
        "selection_note": "Only completed support is used; long-running cases can be underrepresented. A survival/censoring extension is not implemented.",
        "simulator_limit": "One case occupies one department server; staged graph controls service duration. Not a full cross-department event simulator.",
        "query_isolation_required": True,
        "known_simulator_causality_only": True,
    }
    return X, y, meta


def sample_episode(
    seed: int, split: str = "train", family: str | None = None,
    n_support: int = 128, n_query: int = 64, n_features: int = 16,
    problem_type: str | None = None,
) -> dict[str, Any]:
    """Draw an independent world with a completed context and prediction queries.

    Returned dictionary keys are X_support, y_support, X_query, y_query, metadata.
    X is float32, with NaN missing values; binary labels are int64 and regression
    labels float32.  Preprocessing must be fitted exclusively on support arrays.
    ``test_ood`` forbids train mechanisms and ``train`` forbids OOD mechanisms.
    Metadata includes generator truth for auditing, and MUST NOT be fed to a
    predictor except the declared feature schema / categorical indices.
    """
    if split not in SPLIT_FAMILIES:
        raise ValueError(f"Unknown split {split!r}; choose {tuple(SPLIT_FAMILIES)}")
    if n_support < 2 or n_query < 1 or n_features < 2:
        raise ValueError("Need n_support >= 2, n_query >= 1, n_features >= 2")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    # Independent selector and world streams make an implicit selection exactly
    # reproducible when the same resolved family/type are passed explicitly.
    selector_family = np.random.default_rng(np.random.SeedSequence([int(seed), _SPLIT_NAMESPACE[split], 101]))
    selector_type = np.random.default_rng(np.random.SeedSequence([int(seed), _SPLIT_NAMESPACE[split], 102]))
    family = family or str(selector_family.choice(SPLIT_FAMILIES[split]))
    if family not in SPLIT_FAMILIES[split]:
        raise ValueError(f"Family {family!r} is prohibited in split {split!r}")
    problem_type = problem_type or ("classification" if selector_type.random() < 0.5 else "regression")
    if problem_type not in ("classification", "regression"):
        raise ValueError("problem_type must be classification or regression")
    world_identity = f"{GENERATOR_VERSION}:{split}:{family}:{int(seed)}:{n_support}:{n_query}:{n_features}"
    world_hash = hashlib.sha256(world_identity.encode()).hexdigest()
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), _SPLIT_NAMESPACE[split], int(world_hash[:8], 16)]))
    if family.startswith("workflow_"):
        X, y, detail = _workflow_world(rng, family, n_support, n_query, n_features, problem_type)
    else:
        X, y, detail = _generic_world(rng, family, n_support + n_query, n_features, problem_type)
    identity = f"{world_identity}:{problem_type}"
    metadata = {
        "generator_version": GENERATOR_VERSION,
        "provenance": "Generated locally by the Seeyon Tab synthetic episode factory.",
        "episode_id": hashlib.sha256(identity.encode()).hexdigest()[:20],
        "enterprise_or_world_id": world_hash[:20],
        "seed": int(seed), "split": split, "mechanism_family": family,
        "problem_type": problem_type,
        "n_support": n_support, "n_query": n_query, "n_features": n_features,
        "label_count": 2 if problem_type == "classification" else None,
        "split_guarantee": "Disjoint random-world namespaces; OOD test mechanism families excluded from training and validation.",
        **detail,
    }
    return {
        "X_support": X[:n_support], "y_support": y[:n_support],
        "X_query": X[n_support:], "y_query": y[n_support:],
        "metadata": metadata,
    }
