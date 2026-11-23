# Checkpoints and continuation

A run's `latest.json` points to an immutable checkpoint in `checkpoints/`. The trainer writes each checkpoint to a temporary directory, renames it atomically and then replaces the pointer. An OS lock permits one writer per run directory.

| File | Contents |
| --- | --- |
| `weights.npz` | Canonical float32 weights used by both backends |
| `optimizer.npz` | AdamW first and second moments and the completed update count |
| `metadata.json` | Architecture, resolved config, file checksums, training contract, episode cursor and data fingerprint |

The loader checks file hashes, array names, shapes, dtypes and state consistency. NPZ files are loaded with `allow_pickle=False`.

## Resume a run

```bash
seeyon-tab train --resume runs/pilot --output runs/pilot --max-updates 20
```

`--max-updates` bounds the additional updates in this invocation. Resume restores the original learning-rate horizon, optimizer state and next training episode. A change to the device, backend or wall-clock budget is allowed. Ctrl+C or a file named `STOP` in the run directory requests a checkpoint after the current update. Remove the stop file before continuing.

The data fingerprint includes the exact generator, sampler, preprocessor and corpus-reader source bytes, plus the NumPy version. Disk-backed training also fingerprints the manifest and training arrays. Even an edit to generator provenance text changes this fingerprint. Use the same source revision and data for full-state continuation. To branch from an earlier checkpoint, pass its directory to `--resume` and select a fresh output directory.

## Start from weights

For a new run with the same architecture, initialize from a checkpoint explicitly.

```bash
seeyon-tab train --config configs/mac_pilot.json --init-from runs/pilot --output runs/new-pilot --max-updates 20
```

This copies the weights and starts a new optimizer, schedule and episode sequence. Evaluation and inference can read checkpoints with matching architecture and format from a different source revision. Hardware differences can produce numerical differences across training trajectories.
