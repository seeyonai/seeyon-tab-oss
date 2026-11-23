"""Framework-neutral support-only preprocessing; query labels never shape inputs."""
import numpy as np

def prepare_arrays(xs, ys, xq, yq=None, categorical_indices=(), task="regression", category_vocab=66):
    xs=np.asarray(xs,dtype=np.float32); xq=np.asarray(xq,dtype=np.float32)
    ys=np.asarray(ys,dtype=np.float32).reshape(-1)
    yq=None if yq is None else np.asarray(yq,dtype=np.float32).reshape(-1)
    if xs.ndim!=2 or xq.ndim!=2 or xs.shape[1]!=xq.shape[1] or not len(xs) or not len(xq):
        raise ValueError("Support and query must be nonempty 2D tables with the same feature count")
    if len(ys)!=len(xs) or (yq is not None and len(yq)!=len(xq)):
        raise ValueError("Feature and target row counts differ")
    if not np.isfinite(ys).all() or (yq is not None and not np.isfinite(yq).all()):
        raise ValueError("Missing/censored labels need a separate objective")
    if np.isinf(xs).any() or np.isinf(xq).any():
        raise ValueError("Infinite features are invalid; encode missing values as NaN")
    if task not in ("classification","regression") or category_vocab<3:
        raise ValueError("Invalid task or category vocabulary")
    f=xs.shape[1]; cat=np.zeros(f,dtype=bool)
    for j in categorical_indices:
        if type(j) not in (int,np.int32,np.int64) or j<0 or j>=f:
            raise ValueError("Invalid categorical index")
        cat[j]=True
    x=np.concatenate((xs,xq)); missing=np.isnan(x)
    numeric=np.zeros(x.shape,dtype=np.float32); categories=np.zeros(x.shape,dtype=np.int32)
    means=np.zeros(f,dtype=np.float32); scales=np.ones(f,dtype=np.float32)
    for j in range(f):
        observed=xs[np.isfinite(xs[:,j]),j]
        if cat[j]:
            mapping={float(v):i+2 for i,v in enumerate(np.unique(observed)[:category_vocab-2])}
            categories[:,j]=[1 if np.isnan(v) else mapping.get(float(v),0) for v in x[:,j]]
        else:
            if len(observed):
                means[j]=observed.mean(dtype=np.float64)
                scales[j]=max(float(observed.std(dtype=np.float64)),1e-5)
            valid=~missing[:,j]
            numeric[valid,j]=np.clip((x[valid,j]-means[j])/scales[j],-12,12)
    target_mean=0.0; target_scale=1.0
    if task=="regression":
        target_mean=float(ys.mean(dtype=np.float64)); target_scale=max(float(ys.std(dtype=np.float64)),1e-5)
        ys=(ys-target_mean)/target_scale
        if yq is not None: yq=(yq-target_mean)/target_scale
    elif not np.isin(ys,[0,1]).all() or (yq is not None and not np.isin(yq,[0,1]).all()):
        raise ValueError("Prototype supports binary classification")
    batch={"numeric":numeric,"categories":categories,"missing":missing,"is_categorical":cat,
           "y_support":ys.astype(np.float32),"task":task,
           "target_mean":target_mean,"target_scale":target_scale,
           "feature_mean":means,"feature_scale":scales}
    if yq is not None: batch["targets"]=yq.astype(np.float32)
    return batch

def prepare_episode(episode, category_vocab=66):
    meta=episode["metadata"]
    return prepare_arrays(episode["X_support"],episode["y_support"],episode["X_query"],
                          episode.get("y_query"),meta["categorical_indices"],
                          meta["problem_type"],category_vocab)

