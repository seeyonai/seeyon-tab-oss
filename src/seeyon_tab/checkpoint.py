"""Portable, checksummed, atomic-directory checkpoints. No pickle or executable payload."""
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .config import ARCHITECTURE, canonical_json, validate_config, contract_digest
from .weights import validate_weights, weights_digest

FORMAT = "seeyon-tab-npz-v1"

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    with tmp.open("w",encoding="utf-8") as handle:
        handle.write(json.dumps(value,indent=2,allow_nan=False)+"\n")
        handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp,path)

def _numeric_arrays(arrays):
    result={}
    for key,value in arrays.items():
        a=np.asarray(value)
        if not isinstance(key,str) or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
            raise ValueError("Checkpoint arrays must be finite numeric arrays with string names")
        result[key]=a
    return result

def validate_optimizer_state(state, weights, step):
    expected={"step"} | {"m::"+k for k in weights} | {"v::"+k for k in weights}
    if set(state)!=expected:
        raise ValueError("Optimizer keys differ from canonical weights")
    scalar=np.asarray(state["step"])
    if scalar.shape!=() or scalar.dtype.kind not in "iu" or int(scalar)!=step or step<0:
        raise ValueError("Optimizer step disagrees with trainer step")
    for key,value in weights.items():
        for prefix in ("m::","v::"):
            a=np.asarray(state[prefix+key])
            if a.shape!=value.shape or a.dtype!=np.float32 or not np.isfinite(a).all():
                raise ValueError("Invalid AdamW moment "+prefix+key)
            if prefix=="v::" and np.any(a<0):
                raise ValueError("Negative AdamW second moment")
    return state

def save_checkpoint(path, weights, optimizer, state):
    path=Path(path)
    if path.exists(): raise FileExistsError(f"Checkpoint already exists: {path}")
    config=validate_config(state["config"])
    validate_weights(weights,config["model"])
    step=state["step"]
    if type(step) is not int or step<0 or state["next_episode"] != step*config["training"]["accumulation"]:
        raise ValueError("Trainer step/cursor is inconsistent")
    validate_optimizer_state(optimizer,weights,step)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix="."+path.name+"-",dir=path.parent))
    try:
        np.savez_compressed(temp/"weights.npz",**_numeric_arrays(weights))
        np.savez_compressed(temp/"optimizer.npz",**_numeric_arrays(optimizer))
        meta={"format":FORMAT,"architecture":ARCHITECTURE,"created_at":datetime.now(timezone.utc).isoformat(),
              "config":config,"contract_sha256":contract_digest(config),
              "weights_digest":weights_digest(weights),"trainer":{k:v for k,v in state.items() if k!="config"},
              "files":{name:sha256(temp/name) for name in ("weights.npz","optimizer.npz")}}
        atomic_json(temp/"metadata.json",meta)
        os.replace(temp,path)
    finally:
        if temp.exists(): shutil.rmtree(temp)
    return path

def resolve_checkpoint(path):
    path=Path(path).resolve()
    if (path/"latest.json").is_file():
        rel=json.loads((path/"latest.json").read_text())["checkpoint"]
        candidate=(path/rel).resolve()
        if not candidate.is_relative_to(path):
            raise ValueError("Checkpoint pointer escapes run directory")
        return candidate
    return path

def load_checkpoint(path):
    path=resolve_checkpoint(path)
    meta=json.loads((path/"metadata.json").read_text(encoding="utf-8"))
    if meta.get("format")!=FORMAT or meta.get("architecture")!=ARCHITECTURE:
        raise ValueError("Unsupported checkpoint format/architecture")
    config=validate_config(meta["config"])
    if contract_digest(config)!=meta.get("contract_sha256"):
        raise ValueError("Checkpoint training contract digest differs")
    if set(meta.get("files",{}))!={"weights.npz","optimizer.npz"}:
        raise ValueError("Checkpoint file list differs")
    for name,digest in meta["files"].items():
        if sha256(path/name)!=digest: raise ValueError("Checkpoint checksum failed: "+name)
    with np.load(path/"weights.npz",allow_pickle=False) as f:
        weights={k:f[k].copy() for k in f.files}
    with np.load(path/"optimizer.npz",allow_pickle=False) as f:
        optimizer={k:f[k].copy() for k in f.files}
    validate_weights(weights,config["model"])
    if weights_digest(weights)!=meta.get("weights_digest"):
        raise ValueError("Weights content digest differs")
    trainer=meta["trainer"]
    if type(trainer["step"]) is not int or trainer["next_episode"]!=trainer["step"]*config["training"]["accumulation"]:
        raise ValueError("Checkpoint episode cursor disagrees with step")
    validate_optimizer_state(optimizer,weights,trainer["step"])
    return weights,optimizer,meta

