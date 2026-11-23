"""End-to-end controller resume, log recovery and run-lineage protection."""
import contextlib
import importlib.util
import io
import fcntl
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from seeyon_tab.config import validate_config
from seeyon_tab.checkpoint import load_checkpoint, save_checkpoint, atomic_json
from seeyon_tab.trainer import train
from seeyon_tab.weights import weights_digest

@unittest.skipUnless(importlib.util.find_spec("torch"),"PyTorch not installed")
class TrainerTests(unittest.TestCase):
    def config(self):
        return validate_config({"backend":"torch","device":"cpu",
            "model":{"width":16,"heads":2,"row_layers":1,"icl_layers":1,"max_features":16},
            "episode":{"support":8,"query":4,"features":4},
            "training":{"steps":4,"warmup_steps":1,"accumulation":2},
            "runtime":{"threads":1,"save_every":1,"log_every":100,"max_seconds":60}})
    def run_train(self,*a,**k):
        with contextlib.redirect_stdout(io.StringIO()): return train(*a,**k)
    def test_paused_controller_replays_uninterrupted_run(self):
        c=self.config()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); full=root/"full"; paused=root/"paused"
            self.run_train(c,full)
            report=self.run_train(c,paused,max_updates=2)
            self.assertEqual(report["status"],"paused")
            self.assertEqual(report["next_episode"],4)
            resumed=self.run_train(c,paused,resume=paused)
            self.assertEqual(resumed["global_step"],4)
            a=load_checkpoint(full)[0]; b=load_checkpoint(paused)[0]
            self.assertEqual(weights_digest(a),weights_digest(b))
            logs=[json.loads(x) for x in (paused/"train.jsonl").read_text().splitlines()]
            self.assertEqual([r["step"] for r in logs],[1,2,3,4])
    def test_concurrent_writer_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with (root/".run.lock").open("a+") as lock:
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError,"active"):
                    self.run_train(self.config(),root)
                fcntl.flock(lock.fileno(),fcntl.LOCK_UN)

    def test_old_checkpoint_cannot_mix_run_history(self):
        c=self.config()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"run"; self.run_train(c,root,max_updates=2)
            with self.assertRaisesRegex(ValueError,"latest"):
                self.run_train(c,root,resume=root/"checkpoints"/"step_00000001")
            with self.assertRaises(FileExistsError): self.run_train(c,root)
    def test_orphan_checkpoint_requires_matching_optimizer(self):
        c=self.config()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"run"; self.run_train(c,root,max_updates=2)
            cp=root/"checkpoints"/"step_00000002"
            weights,optimizer,meta=load_checkpoint(cp)
            key=next(k for k in optimizer if k.startswith("m::"))
            optimizer[key].flat[0]+=1
            shutil.rmtree(cp)
            save_checkpoint(cp,weights,optimizer,{"config":meta["config"],**meta["trainer"]})
            atomic_json(root/"latest.json",{"checkpoint":"checkpoints/step_00000001","step":1})
            with self.assertRaisesRegex(FileExistsError,"different history"):
                self.run_train(c,root,resume=root,max_updates=1)
            self.assertEqual(load_checkpoint(root)[2]["trainer"]["step"],1)
    def test_uncommitted_logs_archived_and_stop_file_respected(self):
        c=self.config()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"run"; self.run_train(c,root,max_updates=2)
            with (root/"train.jsonl").open("a") as f:
                f.write('{"step":3,"uncommitted":true}\n{"step":')
            self.run_train(c,root,resume=root,max_updates=1)
            self.assertEqual(len(list((root/"recovery").glob("*.jsonl"))),1)
            logs=[json.loads(x) for x in (root/"train.jsonl").read_text().splitlines()]
            self.assertEqual([r["step"] for r in logs],[1,2,3])
            (root/"STOP").touch()
            stopped=self.run_train(c,root,resume=root)
            self.assertEqual(stopped["stop_reason"],"stop_file")
            self.assertEqual(stopped["session_updates"],0)

if __name__=="__main__": unittest.main()

