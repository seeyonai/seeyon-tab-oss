"""Predict unlabeled NumPy query rows with a portable checkpoint."""
import argparse
import numpy as np
from seeyon_tab.backends import create_backend
from seeyon_tab.checkpoint import load_checkpoint
from seeyon_tab.data import sample_episode, prediction_view
from seeyon_tab.preprocess import prepare_episode

p=argparse.ArgumentParser()
p.add_argument("--checkpoint",required=True)
p.add_argument("--backend",choices=["mlx","torch"],default="mlx")
p.add_argument("--device",default="gpu")
args=p.parse_args()
weights,_,metadata=load_checkpoint(args.checkpoint)
config=metadata["config"]
backend=create_backend(args.backend,config["model"],weights,args.device,config["optimizer"])
raw=sample_episode(150,split="validation",family="workflow_parallel",
                   n_support=64,n_query=4,n_features=16,problem_type="regression")
batch=prepare_episode(prediction_view(raw),config["model"]["category_vocab"])
assert "targets" not in batch
prediction=backend.predict(batch)*batch["target_scale"]+batch["target_mean"]
print("Illustrative uncalibrated 10th/50th/90th percentile predictions:")
print(prediction)

