"""Behavioral checks for data visibility, config, schedule and checkpoint integrity."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from seeyon_tab.config import validate_config, contract_digest
from seeyon_tab.weights import initialize_weights, weights_digest
from seeyon_tab.preprocess import prepare_arrays,prepare_episode
from seeyon_tab.checkpoint import save_checkpoint,load_checkpoint
from seeyon_tab.sampling import EpisodeSource
from seeyon_tab.trainer import learning_rate
from seeyon_tab.metrics import score_episode
from seeyon_tab.data import sample_episode

class CommonTests(unittest.TestCase):
    def config(self):
        return validate_config({"backend":"torch","device":"cpu",
            "model":{"width":16,"heads":2,"row_layers":1,"icl_layers":1},
            "training":{"steps":10,"warmup_steps":2,"accumulation":2}})
    def test_invalid_config_fails_early(self):
        for bad in ({"surprise":2},{"model":{"heads":3}},
                    {"runtime":{"max_seconds":float("nan")}},
                    {"optimizer":{"eps":float("inf")}},
                    {"data":{"families":["workflow_rework"]}}):
            with self.assertRaises(ValueError): validate_config(bad)
    def test_device_change_preserves_training_contract(self):
        a=self.config(); b=copy.deepcopy(a)
        b["backend"]="mlx"; b["device"]="gpu"
        self.assertEqual(contract_digest(a),contract_digest(b))
        b["training"]["steps"]+=1
        self.assertNotEqual(contract_digest(a),contract_digest(b))
    def test_target_independence_and_unknown_categories(self):
        xs=np.array([[1,5],[3,7],[np.nan,5]],np.float32)
        xq=np.array([[10000,999],[np.nan,np.nan]],np.float32)
        a=prepare_arrays(xs,[1,2,3],xq,[4,5],[1])
        b=prepare_arrays(xs,[1,2,3],xq,[-999,999],[1])
        self.assertEqual(a["feature_mean"][0],2)
        np.testing.assert_array_equal(a["numeric"],b["numeric"])
        self.assertEqual(a["categories"][-2,1],0)
        self.assertEqual(a["categories"][-1,1],1)
        self.assertNotIn("targets",prepare_arrays(xs,[1,2,3],xq,categorical_indices=[1]))
    def test_episode_cursor_and_learning_rate_replay(self):
        c=self.config(); a=EpisodeSource(c); b=EpisodeSource(c)
        ids=[a.get(i)["metadata"]["episode_id"] for i in range(8)]
        self.assertEqual(ids[4:],[b.get(i)["metadata"]["episode_id"] for i in range(4,8)])
        self.assertLess(learning_rate(c,0),learning_rate(c,1))
        self.assertAlmostEqual(learning_rate(c,9),c["training"]["learning_rate"]*c["training"]["min_lr_fraction"])
    def test_checkpoint_checksum_and_cursor_integrity(self):
        c=self.config(); w=initialize_weights(c["model"],3)
        opt={"step":np.array(2,dtype=np.int64)}
        opt.update({"m::"+k:np.zeros_like(v) for k,v in w.items()})
        opt.update({"v::"+k:np.zeros_like(v) for k,v in w.items()})
        state={"config":c,"step":2,"next_episode":4}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"ckpt"; save_checkpoint(path,w,opt,state)
            restored,_,_=load_checkpoint(path)
            self.assertEqual(weights_digest(w),weights_digest(restored))
            with self.assertRaises(FileExistsError): save_checkpoint(path,w,opt,state)
            with (path/"weights.npz").open("ab") as f: f.write(b"corruption")
            with self.assertRaisesRegex(ValueError,"checksum"): load_checkpoint(path)
        with tempfile.TemporaryDirectory() as d:
            bad=dict(state,next_episode=5)
            with self.assertRaises(ValueError): save_checkpoint(Path(d)/"x",w,opt,bad)
    def test_scaled_metrics_and_prediction_shape(self):
        raw=sample_episode(19,split="validation",family="linear",n_support=12,n_query=5,n_features=4,problem_type="regression")
        prepared=prepare_episode(raw); y=prepared["targets"]
        pred=np.stack((y-1,y,y+1),axis=1)
        score=score_episode(pred,raw,prepared)
        self.assertLess(score["mae_support_scaled"],1e-6)
        self.assertEqual(score["interval_80_coverage"],1)
        with self.assertRaises(ValueError): score_episode(pred[:,:2],raw,prepared)

if __name__=="__main__": unittest.main()

