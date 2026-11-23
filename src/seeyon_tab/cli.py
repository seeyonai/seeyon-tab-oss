"""Command line for setup checks, training, evaluation, parity and data audit."""
import argparse
import json
from pathlib import Path
from .config import load_config, validate_config
from .checkpoint import atomic_json, load_checkpoint

def main(argv=None):
    parser=argparse.ArgumentParser(prog="seeyon-tab")
    commands=parser.add_subparsers(dest="command",required=True)
    d=commands.add_parser("doctor")
    d.add_argument("--backend",choices=["mlx","torch"],default="mlx")
    d.add_argument("--device",default="gpu"); d.add_argument("--output")
    t=commands.add_parser("train")
    t.add_argument("--config"); t.add_argument("--output",required=True)
    t.add_argument("--resume"); t.add_argument("--init-from")
    t.add_argument("--backend",choices=["mlx","torch"]); t.add_argument("--device")
    t.add_argument("--max-updates",type=int); t.add_argument("--max-seconds",type=float)
    e=commands.add_parser("evaluate")
    e.add_argument("--checkpoint",required=True)
    e.add_argument("--split",choices=["validation","test_iid","test_ood"],default="validation")
    e.add_argument("--backend",choices=["mlx","torch"]); e.add_argument("--device")
    e.add_argument("--episodes",type=int); e.add_argument("--corpus")
    e.add_argument("--output",required=True)
    p=commands.add_parser("parity")
    p.add_argument("--checkpoint"); p.add_argument("--mlx-device",default="cpu")
    p.add_argument("--torch-device",default="cpu"); p.add_argument("--output")
    a=commands.add_parser("audit-data"); a.add_argument("--corpus",default="data/pilot"); a.add_argument("--output")
    i=commands.add_parser("inspect"); i.add_argument("--config")
    i.add_argument("--checkpoint")
    args=parser.parse_args(argv)
    try:
        if args.command=="doctor":
            from .doctor import doctor
            result=doctor(args.backend,args.device)
            if args.output: atomic_json(args.output,result)
            print(json.dumps(result,indent=2)); return 0 if result["ok"] else 1
        if args.command=="train":
            from .trainer import train
            if args.config: config=load_config(args.config)
            elif args.resume: config=load_checkpoint(args.resume)[2]["config"]
            else: parser.error("train needs --config or --resume")
            if args.backend: config["backend"]=args.backend
            if args.device: config["device"]=args.device
            train(config,args.output,args.resume,args.init_from,args.max_updates,args.max_seconds)
            return 0
        if args.command=="evaluate":
            from .trainer import evaluate_checkpoint
            result=evaluate_checkpoint(args.checkpoint,args.split,args.backend,args.device,args.episodes,args.corpus)
            atomic_json(args.output,result)
            print(json.dumps(result["summary"],indent=2)); return 0
        if args.command=="parity":
            from .parity import run_parity
            result=run_parity(checkpoint=args.checkpoint,mlx_device=args.mlx_device,
                              torch_device=args.torch_device,output=args.output)
            print(json.dumps(result,indent=2)); return 0
        if args.command=="audit-data":
            from .data import EpisodeCorpus
            result={}
            for split in ("train","validation","test_iid","test_ood"):
                corpus=EpisodeCorpus(args.corpus,split)
                count=rows=0
                for episode in corpus:
                    count+=1; rows+=len(episode["y_support"])+len(episode["y_query"])
                result[split]={"episodes":count,"rows":rows}
            if args.output: atomic_json(args.output,result)
            print(json.dumps(result,indent=2)); return 0
        if args.command=="inspect":
            from .weights import parameter_count
            if args.checkpoint:
                _,_,result=load_checkpoint(args.checkpoint)
            elif args.config:
                config=load_config(args.config)
                result={"config":config,"parameters":parameter_count(config["model"])}
            else: parser.error("inspect needs --config or --checkpoint")
            print(json.dumps(result,indent=2)); return 0
    except (ValueError,RuntimeError,FileNotFoundError,FileExistsError,ImportError) as exc:
        parser.exit(2,f"{type(exc).__name__}: {exc}\n")

if __name__=="__main__": raise SystemExit(main())

