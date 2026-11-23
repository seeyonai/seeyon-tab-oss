# Seeyon Tab

Seeyon Tab trains a tabular in-context predictor from random initialization. Give it labeled support rows and unlabeled query rows from the same table. It returns two class logits per query for binary classification, or the 10th, 50th and 90th percentiles for regression.

Training and validation draw reproducible synthetic tables locally with NumPy. The MLX and PyTorch implementations share one checkpoint format. Validation measures performance on the simulated tasks. Performance on real tables requires separate evaluation.

## Install

From the repository root on an Apple Silicon Mac, use a native arm64 Python 3.12 environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[mlx]'
seeyon-tab doctor --backend mlx --device gpu
```

For a CPU run, install `python -m pip install -e '.[torch]' 'torch==2.9.1'` and add `--backend torch --device cpu` to each training, evaluation and prediction command. See the [Metal verification guide](docs/MAC_MLX.md) and [CUDA setup](docs/CUDA_MIGRATION.md) for backend-specific commands.

## Reproduce the pilot checkpoint

The pilot has 722,949 parameters. Each update accumulates four episodes, each with 128 labeled rows, 32 queries and 16 features. The default training seed is `20261123`. These commands replay the first 200 updates of the 2,000-update schedule, with a checkpoint at update 20. The seed fixes the initial weights and episode sequence. Repeated Metal runs can produce slightly different trained weights.

```bash
seeyon-tab train --config configs/mac_pilot.json --output runs/pilot --max-updates 20
seeyon-tab train --resume runs/pilot --output runs/pilot --max-updates 180
seeyon-tab inspect --checkpoint runs/pilot
seeyon-tab evaluate --checkpoint runs/pilot --split validation --episodes 204 --output runs/pilot/validation.json
python examples/predict.py --checkpoint runs/pilot --backend mlx --device gpu
```

At update 200 the next training episode index is 800. To continue the original schedule, run `seeyon-tab train --resume runs/pilot --output runs/pilot`. The per-invocation time budget also applies. Each evaluation replays the same split and seed. The JSON contains episode IDs, per-episode scores and means, including support-only constant baselines for classification and regression.

| Preset | Purpose |
| --- | --- |
| [mac_smoke.json](configs/mac_smoke.json) | 122,501 parameters and 20 updates for a short train/resume check |
| [mac_pilot.json](configs/mac_pilot.json) | Generic table mechanisms and workflow simulations |
| [mac_pilot_generic.json](configs/mac_pilot_generic.json) | Same pilot architecture, seed and schedule, trained on linear, additive and interaction mechanisms |

For a matched comparison, train the generic preset in its own output directory with the same update budget and evaluate both checkpoints on the same validation episodes. For additional replicates, set `training.seed` to `20271123` or `20281123` in copies of both presets and start each run in a fresh directory. Keep the evaluation seed fixed.

The [architecture guide](docs/ARCHITECTURE.md) describes the prediction boundary and synthetic splits. The [checkpoint guide](docs/CHECKPOINTS.md) covers resume, weight initialization and portability. The project uses the [MIT license](LICENSE).
