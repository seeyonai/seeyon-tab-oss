"""Bounded, resumable episodic training shared by MLX and PyTorch."""
import copy
import fcntl
import json
import math
import os
import platform
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from . import __version__
from .backends import create_backend
from .config import validate_config, contract_digest, canonical_json
from .checkpoint import atomic_json, save_checkpoint, load_checkpoint
from .data import GENERATOR_VERSION, EpisodeCorpus
from .weights import initialize_weights, parameter_count, weights_digest
from .preprocess import prepare_episode
from .sampling import EpisodeSource, evaluation_episode
from .metrics import score_episode, summarize

def learning_rate(config, completed_updates):
    t=config["training"]; step=completed_updates; warm=t["warmup_steps"]
    if warm and step<warm: factor=(step+1)/warm
    else:
        progress=(step-warm)/max(1,t["steps"]-warm-1)
        factor=t["min_lr_fraction"]+(1-t["min_lr_fraction"])*.5*(1+math.cos(math.pi*min(1,max(0,progress))))
    return t["learning_rate"]*factor

def evaluate_backend(backend, config, split="validation", episodes=None, corpus=None):
    if split not in ("validation","test_iid","test_ood"):
        raise ValueError("Use a held-out evaluation split")
    count=episodes if episodes is not None else config["evaluation"]["episodes"]
    if type(count) is not int or count<2: raise ValueError("Evaluate at least two episodes")
    dataset=EpisodeCorpus(corpus,split) if corpus else None
    if dataset is not None: count=min(count,len(dataset))
    rows=[]; started=time.perf_counter()
    for i in range(count):
        raw=dataset[i] if dataset is not None else evaluation_episode(config,split,i)
        batch=prepare_episode(raw,config["model"]["category_vocab"])
        predicted=backend.predict(batch)
        rows.append(score_episode(predicted,raw,batch))
    return {"status":"synthetic_research_evaluation","scope":"No real-enterprise or calibrated-performance claim",
            "split":split,"seed":config["evaluation"]["seed"],"source":"corpus" if dataset else "online",
            "generator_version":GENERATOR_VERSION,"episodes":rows,"summary":summarize(rows),
            "seconds":time.perf_counter()-started,"backend":backend.runtime_info(),
            "aggregation":"Equal episode weights. Regression errors scaled by support target standard deviation."}

def _append_json(path,value):
    with Path(path).open("a",encoding="utf-8") as f:
        f.write(json.dumps(value,allow_nan=False)+"\n")

def _recover_training_log(output, committed_step):
    """Archive log entries newer than the durable checkpoint after a hard kill."""
    path=output/"train.jsonl"
    if not path.exists(): return
    kept=[]; tail=[]
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        try:
            row=json.loads(line)
            valid=type(row.get("step")) is int and row["step"]<=committed_step
        except (ValueError,AttributeError): valid=False
        (kept if valid else tail).append(line)
    if tail:
        recovery=output/"recovery"; recovery.mkdir(exist_ok=True)
        archive=recovery/f"uncommitted_{time.time_ns()}.jsonl"
        with archive.open("w",encoding="utf-8") as f:
            f.writelines(tail); f.flush(); os.fsync(f.fileno())
        temporary=path.with_suffix(".jsonl.tmp")
        with temporary.open("w",encoding="utf-8") as f:
            f.writelines(kept); f.flush(); os.fsync(f.fileno())
        os.replace(temporary,path)

def _has_run_content(path):
    return path.exists() and any(p.name!=".run.lock" for p in path.iterdir())

def train(config,output,resume=None,init_from=None,max_updates=None,max_seconds=None):
    """Hold an OS lock for the complete run; kernel releases it after a crash."""
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    with (output/".run.lock").open("a+") as lock:
        try: fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another trainer is active in this output directory") from exc
        try:
            return _train_locked(config,output,resume,init_from,max_updates,max_seconds)
        finally: fcntl.flock(lock.fileno(),fcntl.LOCK_UN)

def _train_locked(config,output,resume=None,init_from=None,max_updates=None,max_seconds=None):
    c=validate_config(config); output=Path(output)
    if resume and init_from: raise ValueError("Choose resume or init-from, not both")
    if max_updates is not None and (type(max_updates) is not int or max_updates<1):
        raise ValueError("max_updates must be positive")
    budget=c["runtime"]["max_seconds"] if max_seconds is None else max_seconds
    if budget is not None and (not math.isfinite(budget) or budget<=0):
        raise ValueError("Time budget must be finite and positive")
    if c["backend"]=="torch":
        import torch
        torch.set_num_threads(c["runtime"]["threads"])
    source=EpisodeSource(c)
    old_metadata=None
    if resume:
        weights,opt_state,old_metadata=load_checkpoint(resume)
        if old_metadata["contract_sha256"]!=contract_digest(c):
            raise ValueError("Resume contract differs (architecture, data, optimizer or schedule). Use identical settings; init-from explicitly starts a new optimizer/schedule.")
        prior=old_metadata["trainer"]
        if prior["source_fingerprint"]!=source.fingerprint:
            raise ValueError("Training corpus/generator changed since checkpoint")
        step=prior["step"]; cursor=prior["next_episode"]
        initial_digest=prior["initial_weights_digest"]
    else:
        step=0; cursor=0
        if init_from:
            weights,_,old_metadata=load_checkpoint(init_from)
            if old_metadata["config"]["model"]!=c["model"]:
                raise ValueError("init-from needs identical model architecture; enlarging a model requires a separate method")
        else: weights=initialize_weights(c["model"],c["training"]["seed"])
        initial_digest=weights_digest(weights)
        opt_state=None
    if step>c["training"]["steps"]: raise ValueError("Checkpoint exceeds configured schedule")
    if _has_run_content(output) and not resume:
        raise FileExistsError("Output is not empty. Select a fresh run directory or resume explicitly.")
    if resume and _has_run_content(output):
        if not (output/"run_manifest.json").is_file() or not (output/"latest.json").is_file():
            raise ValueError("Resume output is nonempty but lacks a complete run manifest/latest pointer; use a fresh directory")
        manifest=json.loads((output/"run_manifest.json").read_text())
        if (manifest["contract_sha256"]!=contract_digest(c) or
            manifest["initial_weights_digest"]!=initial_digest or
            manifest["source_fingerprint"]!=source.fingerprint):
            raise ValueError("Output directory belongs to another training contract or weight lineage")
        existing_weights,existing_optimizer,existing_meta=load_checkpoint(output)
        if (existing_meta["trainer"]["step"]!=step or
            weights_digest(existing_weights)!=weights_digest(weights) or
            any(not np.array_equal(existing_optimizer[k],opt_state[k]) for k in existing_optimizer)):
            raise ValueError("Requested resume does not match this run's latest checkpoint; use a fresh output directory for a branch")
    output.mkdir(parents=True,exist_ok=True)
    if resume: _recover_training_log(output,step)
    backend=create_backend(c["backend"],c["model"],weights,c["device"],c["optimizer"])
    if opt_state is not None: backend.set_optimizer_state(opt_state)
    if not (output/"run_manifest.json").exists():
        atomic_json(output/"run_manifest.json",{
            "created_at":datetime.now(timezone.utc).isoformat(),"package_version":__version__,
            "config":c,"contract_sha256":contract_digest(c),"source_fingerprint":source.fingerprint,
            "parameters":parameter_count(c["model"]),"initial_weights_digest":initial_digest,
            "origin_checkpoint":str(resume or init_from) if (resume or init_from) else None,
        })
    atomic_json(output/"resolved_config.json",c)
    stop={"requested":False,"signal":None}
    def handler(signum,frame):
        stop["requested"]=True; stop["signal"]=signum
    previous={}
    for sig in (signal.SIGINT,signal.SIGTERM):
        previous[sig]=signal.signal(sig,handler)
    durations=[]; generated_seconds=0.0; session_updates=0; started=time.perf_counter()
    last_saved=-1; reason="complete"
    def checkpoint():
        nonlocal last_saved
        if last_saved==step: return
        relative=Path("checkpoints")/f"step_{step:08d}"
        destination=output/relative
        state={"config":c,"step":step,"next_episode":cursor,"source_fingerprint":source.fingerprint,
               "initial_weights_digest":initial_digest,"generator_version":GENERATOR_VERSION,
               "backend":backend.runtime_info(),"package_version":__version__}
        if destination.exists():
            # An orphan checkpoint may predate latest.json after a hard kill.
            # Equal weights alone do not establish equal continuation state.
            existing,existing_opt,meta=load_checkpoint(destination)
            live_opt=backend.get_optimizer_state()
            if (weights_digest(existing)!=weights_digest(backend.export_weights()) or
                meta["contract_sha256"]!=contract_digest(c) or
                any(meta["trainer"][key]!=state[key] for key in
                    ("step","next_episode","source_fingerprint","initial_weights_digest")) or
                any(not np.array_equal(existing_opt[key],live_opt[key]) for key in live_opt)):
                raise FileExistsError("A different history already occupies this checkpoint step; use a fresh output directory")
        else:
            save_checkpoint(destination,backend.export_weights(),backend.get_optimizer_state(),state)
        atomic_json(output/"latest.json",{"checkpoint":relative.as_posix(),"step":step})
        last_saved=step
    try:
        while step<c["training"]["steps"]:
            if stop["requested"]: reason="signal"; break
            if (output/"STOP").exists(): reason="stop_file"; break
            if max_updates is not None and session_updates>=max_updates: reason="update_budget"; break
            if budget is not None and time.perf_counter()-started>=budget: reason="time_budget"; break
            update_start=time.perf_counter(); gen_start=update_start
            raw_episodes=[source.get(cursor+i) for i in range(c["training"]["accumulation"])]
            if any(ep["metadata"]["split"]!="train" for ep in raw_episodes):
                raise ValueError("Refusing evaluation data in a training update")
            batches=[prepare_episode(ep,c["model"]["category_vocab"]) for ep in raw_episodes]
            gen_elapsed=time.perf_counter()-gen_start; generated_seconds+=gen_elapsed
            lr=learning_rate(c,step); metrics=backend.train_step(batches,lr)
            backend.synchronize()
            step+=1; cursor+=len(batches); session_updates+=1
            seconds=time.perf_counter()-update_start; durations.append(seconds)
            log={"step":step,"next_episode":cursor,"learning_rate":lr,
                 "loss":float(metrics["loss"]),"grad_norm":float(metrics["grad_norm"]),
                 "seconds":seconds,"data_seconds":gen_elapsed,
                 "backend":c["backend"],"device":c["device"],
                 "episode_ids":[ep["metadata"]["episode_id"] for ep in raw_episodes]}
            _append_json(output/"train.jsonl",log)
            if session_updates==1 or step%c["runtime"]["log_every"]==0:
                print(json.dumps({k:log[k] for k in ("step","loss","grad_norm","seconds")}),flush=True)
            if step%c["runtime"]["save_every"]==0: checkpoint()
            interval=c["runtime"]["eval_every"]
            if interval and step%interval==0:
                evaluation=evaluate_backend(backend,c)
                atomic_json(output/"validation"/f"step_{step:08d}.json",evaluation)
        checkpoint()
        profiled=durations[min(5,max(0,len(durations)-1)):]
        elapsed=time.perf_counter()-started
        report={"status":"completed_schedule" if step==c["training"]["steps"] else "paused",
                "stop_reason":reason,"global_step":step,"next_episode":cursor,
                "session_updates":session_updates,"session_episode_draws":session_updates*c["training"]["accumulation"],
                "parameters":parameter_count(c["model"]),"runtime":backend.runtime_info(),
                "session_seconds_including_generation_and_checkpoints":elapsed,
                "generation_and_preprocess_seconds":generated_seconds,
                "median_update_seconds_after_warmup":float(np.median(profiled)) if profiled else None,
                "p95_update_seconds_after_warmup":float(np.quantile(profiled,.95)) if profiled else None,
                "memory":backend.memory_stats(),"final_weights_digest":weights_digest(backend.export_weights()),
                "initial_weights_digest":initial_digest,"contract_sha256":contract_digest(c),
                "warning":"Engineering/research run; no production quality or real-enterprise validation implied."}
        _append_json(output/"sessions.jsonl",report)
        atomic_json(output/"training_report.json",report)
        print(json.dumps({"status":report["status"],"step":step,"checkpoint_run":str(output)}),flush=True)
        return report
    finally:
        for sig,prior_handler in previous.items(): signal.signal(sig,prior_handler)

def evaluate_checkpoint(checkpoint,split="validation",backend_name=None,device=None,episodes=None,corpus=None):
    weights,_,meta=load_checkpoint(checkpoint); c=meta["config"]
    name=backend_name or c["backend"]; dev=device or c["device"]
    engine=create_backend(name,c["model"],weights,dev,c["optimizer"])
    result=evaluate_backend(engine,c,split,episodes,corpus)
    result["checkpoint_step"]=meta["trainer"]["step"]
    result["weights_digest"]=meta["weights_digest"]
    return result

