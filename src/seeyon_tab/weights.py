"""Canonical float32 parameters: same names/shapes/orientation in both engines."""
import hashlib
import numpy as np
from .config import model_config

def parameter_shapes(config):
    c=model_config(config); d=c["width"]; f=d*c["ff_multiplier"]
    out={}
    def linear(name, ni, no):
        out[name+".weight"]=(no,ni); out[name+".bias"]=(no,)
    def norm(name):
        out[name+".weight"]=(d,); out[name+".bias"]=(d,)
    linear("numeric_in",1,d); linear("numeric_out",d,d)
    for name,n in (("category_embedding",c["category_vocab"]),
                   ("feature_embedding",c["max_features"]),("type_embedding",2),
                   ("missing_embedding",2),("task_embedding",2),("class_label_embedding",2)):
        out[name+".weight"]=(n,d)
    out["row_token"]=(d,); out["unknown_label"]=(d,)
    linear("reg_label_in",1,d); linear("reg_label_out",d,d)
    for stage,depth in (("row_blocks",c["row_layers"]),("icl_blocks",c["icl_layers"])):
        for i in range(depth):
            prefix=f"{stage}.{i}"
            norm(prefix+".norm1"); norm(prefix+".norm2")
            for proj in ("q_proj","k_proj","v_proj","out_proj"):
                linear(prefix+"."+proj,d,d)
            linear(prefix+".ff1",d,f); linear(prefix+".ff2",f,d)
    norm("output_norm"); linear("class_head",d,2); linear("reg_head",d,3)
    return out

def initialize_weights(config, seed=0):
    rng=np.random.default_rng(seed); result={}
    for name,shape in parameter_shapes(config).items():
        if name.endswith(".bias"):
            value=np.zeros(shape)
        elif ".norm" in name or name.startswith("output_norm"):
            value=np.ones(shape)
        elif "embedding" in name or name in ("row_token","unknown_label"):
            value=rng.normal(0,0.02,size=shape)
        else:
            bound=np.sqrt(6.0/(shape[0]+shape[1]))
            value=rng.uniform(-bound,bound,size=shape)
        result[name]=np.asarray(value,dtype=np.float32)
    return result

def validate_weights(weights,config):
    expected=parameter_shapes(config)
    if set(weights)!=set(expected):
        raise ValueError(f"Parameter keys differ: missing={sorted(set(expected)-set(weights))}, extra={sorted(set(weights)-set(expected))}")
    for name,shape in expected.items():
        value=np.asarray(weights[name])
        if value.shape!=shape or value.dtype!=np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Invalid parameter {name}: expected finite float32 {shape}")
    return weights

def parameter_count(config):
    return sum(int(np.prod(shape)) for shape in parameter_shapes(config).values())

def weights_digest(weights):
    h=hashlib.sha256()
    for key in sorted(weights):
        a=np.ascontiguousarray(weights[key])
        h.update(key.encode()); h.update(str(a.shape).encode()); h.update(a.dtype.str.encode()); h.update(a.tobytes())
    return h.hexdigest()

