"""Strict versioned JSON settings. Unknown keys fail instead of being ignored."""
import copy
import hashlib
import json
import math
from pathlib import Path

ARCHITECTURE = "seeyon-tab-portable-v1"
DEFAULT_MODEL = dict(width=64, heads=4, row_layers=1, icl_layers=2,
                     ff_multiplier=2, max_features=128, category_vocab=66)
DEFAULT_OPTIMIZER = dict(beta1=0.9, beta2=0.999, eps=1e-8,
                         weight_decay=0.01, clip_norm=1.0)
DEFAULT_CONFIG = {
    "schema_version": 1, "name": "smoke", "backend": "mlx", "device": "gpu",
    "model": DEFAULT_MODEL, "optimizer": DEFAULT_OPTIMIZER,
    "episode": {"support": 32, "query": 16, "features": 16},
    "data": {"source": "online", "corpus_path": "data/pilot", "families": []},
    "training": {"steps": 20, "seed": 20261123, "learning_rate": 0.0002,
                 "min_lr_fraction": 0.05, "warmup_steps": 2, "accumulation": 1},
    "evaluation": {"episodes": 12, "seed": 862041},
    "runtime": {"threads": 2, "log_every": 10, "save_every": 10,
                "eval_every": 0, "max_seconds": 600},
}

def _merge(base, supplied, path="config"):
    if not isinstance(supplied, dict):
        raise ValueError(path + " must be an object")
    extra = set(supplied) - set(base)
    if extra:
        raise ValueError(f"Unknown {path} keys: {sorted(extra)}")
    result = copy.deepcopy(base)
    for key, value in supplied.items():
        result[key] = _merge(base[key], value, path+"."+key) if isinstance(base[key], dict) else value
    return result

def model_config(config):
    result = _merge(DEFAULT_MODEL, config, "model")
    for key, value in result.items():
        if type(value) is not int or value < 1:
            raise ValueError("model." + key + " must be a positive integer")
    if result["width"] % result["heads"]:
        raise ValueError("Model width must be divisible by heads")
    if result["category_vocab"] < 3:
        raise ValueError("category_vocab needs unknown, missing and observed slots")
    return result

def validate_config(config):
    c = _merge(DEFAULT_CONFIG, config)
    if c["schema_version"] != 1:
        raise ValueError("Unsupported config schema")
    c["model"] = model_config(c["model"])
    if c["backend"] not in ("mlx", "torch"):
        raise ValueError("backend must be mlx or torch")
    if not isinstance(c["device"], str):
        raise ValueError("device must be a string")
    for group, keys in (("episode", ("support", "query", "features")),
                        ("training", ("steps", "accumulation")),
                        ("evaluation", ("episodes",)),
                        ("runtime", ("threads", "log_every", "save_every"))):
        for k in keys:
            if type(c[group][k]) is not int or c[group][k] < 1:
                raise ValueError(f"{group}.{k} must be positive integer")
    if c["episode"]["support"] < 2 or c["episode"]["features"] < 2:
        raise ValueError("Synthetic episodes need support >=2 and features >=2")
    if c["episode"]["features"] > c["model"]["max_features"]:
        raise ValueError("episode features exceed model maximum")
    for group,key in (("training","seed"),("training","warmup_steps"),("evaluation","seed"),("runtime","eval_every")):
        if type(c[group][key]) is not int or c[group][key] < 0:
            raise ValueError(f"{group}.{key} must be nonnegative integer")
    if c["training"]["warmup_steps"] > c["training"]["steps"]:
        raise ValueError("warmup cannot exceed total steps")
    if not 0 < c["training"]["learning_rate"] < 1:
        raise ValueError("learning_rate must be between zero and one")
    if not 0 < c["training"]["min_lr_fraction"] <= 1:
        raise ValueError("min_lr_fraction must be in (0,1]")
    if c["runtime"]["max_seconds"] is not None and c["runtime"]["max_seconds"] <= 0:
        raise ValueError("max_seconds must be positive or null")
    o=c["optimizer"]
    scalars=list(o.values())+[c["training"]["learning_rate"],c["training"]["min_lr_fraction"]]
    if c["runtime"]["max_seconds"] is not None: scalars.append(c["runtime"]["max_seconds"])
    if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in scalars):
        raise ValueError("Numeric optimizer/schedule/budget settings must be finite")
    if not 0 <= o["beta1"] < 1 or not 0 <= o["beta2"] < 1 or o["eps"] <= 0 or o["clip_norm"] <= 0 or o["weight_decay"] < 0:
        raise ValueError("Invalid AdamW settings")
    if c["data"]["source"] not in ("online","corpus"):
        raise ValueError("data.source must be online or corpus")
    if not isinstance(c["data"]["families"], list):
        raise ValueError("data.families must be a list")
    from .data import SPLIT_FAMILIES
    if not set(c["data"]["families"]).issubset(SPLIT_FAMILIES["train"]):
        raise ValueError("Training configuration contains non-training mechanisms")
    return c

def load_config(path):
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))

def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",",":"), allow_nan=False)

def training_contract(config):
    # Device/backend/runtime can change at migration. Learning task, weights,
    # scheduler horizon and AdamW settings must match for a true resume.
    return {"architecture": ARCHITECTURE,
            **{k:copy.deepcopy(config[k]) for k in ("model","optimizer","episode","data","training")}}

def contract_digest(config):
    return hashlib.sha256(canonical_json(training_contract(config)).encode()).hexdigest()

