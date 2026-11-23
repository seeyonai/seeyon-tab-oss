"""Small NumPy metrics: per-task scale normalization and explicit uncertainty limits."""
import numpy as np

def _binary_ranking(y,p):
    positive=int(y.sum()); negative=len(y)-positive
    if not positive or not negative: return None,None
    # Pairwise definition includes ties, suitable for bounded pilot query sizes.
    pos=p[y==1]; neg=p[y==0]
    auc=float(np.mean((pos[:,None]>neg).astype(float)+0.5*(pos[:,None]==neg)))
    order=np.argsort(-p,kind="stable"); yy=y[order]; pp=p[order]
    boundaries=np.r_[np.where(pp[:-1]!=pp[1:])[0],len(pp)-1]
    tp=np.cumsum(yy)[boundaries]
    recall=tp/positive; precision=tp/(boundaries+1)
    ap=float(np.sum(np.diff(np.r_[0.,recall])*precision))
    return auc,ap

def score_episode(predicted, raw, prepared):
    pred=np.asarray(predicted,dtype=np.float64); y=np.asarray(raw["y_query"],dtype=np.float64)
    if not np.isfinite(pred).all(): raise ValueError("Non-finite predictions")
    meta=raw["metadata"]
    result={"episode_id":meta["episode_id"],"family":meta["mechanism_family"],"task":meta["problem_type"]}
    if result["task"]=="classification":
        if pred.shape!=(len(y),2): raise ValueError("Expected two logits per query")
        e=np.exp(pred-pred.max(axis=1,keepdims=True)); p=e[:,1]/e.sum(axis=1)
        p=np.clip(p,1e-8,1-1e-8)
        prior=(np.sum(raw["y_support"])+1)/(len(raw["y_support"])+2)
        result.update(brier=float(np.mean((p-y)**2)),constant_brier=float(np.mean((prior-y)**2)),
                      log_loss=float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))))
        result["auroc"],result["average_precision"]=_binary_ranking(y,p)
        top=max(1,int(np.ceil(.1*len(y)))); selected=np.argsort(-p,kind="stable")[:top]
        result["precision_at_10pct"]=float(np.mean(y[selected]))
        result["recall_at_10pct"]=None if not y.sum() else float(y[selected].sum()/y.sum())
    else:
        if pred.shape!=(len(y),3) or np.any(np.diff(pred,axis=1)<-1e-6):
            raise ValueError("Expected ordered three-quantile predictions")
        scale=prepared["target_scale"]; quantiles=pred*scale+prepared["target_mean"]
        residual=y[:,None]-quantiles; levels=np.array([.1,.5,.9])
        mae=float(np.mean(np.abs(quantiles[:,1]-y)))
        constant=float(np.mean(np.abs(np.median(raw["y_support"])-y)))
        result.update(mae=mae,constant_mae=constant,mae_support_scaled=mae/scale,
                      constant_mae_support_scaled=constant/scale,
                      pinball_support_scaled=float(np.maximum(levels*residual,(levels-1)*residual).mean()/scale),
                      interval_80_coverage=float(np.mean((y>=quantiles[:,0])&(y<=quantiles[:,2]))),
                      interval_80_width_support_scaled=float(np.mean(quantiles[:,2]-quantiles[:,0])/scale))
    return result

def summarize(rows):
    result={}
    for task in ("classification","regression"):
        selected=[r for r in rows if r["task"]==task]
        if not selected: continue
        keys=(["brier","constant_brier","log_loss","auroc","average_precision","precision_at_10pct","recall_at_10pct"]
              if task=="classification" else ["mae_support_scaled","constant_mae_support_scaled",
                    "pinball_support_scaled","interval_80_coverage","interval_80_width_support_scaled"])
        entry={"episodes":len(selected)}
        for key in keys:
            values=[r[key] for r in selected if r.get(key) is not None]
            entry[key]=float(np.mean(values)) if values else None
            if len(values)!=len(selected): entry[key+"_valid_episodes"]=len(values)
        result[task]=entry
    return result

