"""Read-only runtime and backend checks, including a real tiny forward pass."""
import importlib.metadata
import platform
import sys
import numpy as np
from .backends import create_backend
from .config import DEFAULT_MODEL
from .weights import initialize_weights
from .data import sample_episode
from .preprocess import prepare_episode

def doctor(backend="mlx",device="gpu"):
    versions={}
    for package in ("numpy","mlx","mlx-cpu","torch","seeyon-tab"):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]=None
    report={"python":sys.version.split()[0],"platform":platform.platform(),"machine":platform.machine(),
            "requested_backend":backend,"requested_device":device,"versions":versions,
            "checks":[],"ok":False}
    if platform.system()=="Darwin" and platform.machine()!="arm64":
        report["checks"].append({"ok":False,"message":"Use native arm64 Python; this process appears to run under Rosetta."})
        return report
    try:
        config=dict(DEFAULT_MODEL,width=16,heads=2,row_layers=1,icl_layers=1,max_features=16)
        engine=create_backend(backend,config,initialize_weights(config,4),device)
        raw=sample_episode(8,split="validation",family="linear",n_support=8,n_query=4,n_features=4,problem_type="regression")
        batch=prepare_episode(raw,config["category_vocab"])
        prediction=engine.predict(batch)
        report["checks"].append({"ok":bool(np.isfinite(prediction).all()),"message":"Tiny forward pass completed"})
        report["runtime"]=engine.runtime_info(); report["memory"]=engine.memory_stats()
        report["ok"]=all(c["ok"] for c in report["checks"])
    except Exception as exc:
        report["checks"].append({"ok":False,"message":f"{type(exc).__name__}: {exc}"})
    return report

