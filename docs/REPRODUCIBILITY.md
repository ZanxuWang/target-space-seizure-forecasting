# Reproducibility guide

## Two supported levels

**Paper snapshot (recommended for reviewers).** Recomputes every reported AUC
and every scripted figure from the exact held-out probability/diagnostic arrays
used in the paper. No model weights or GPU are required.

```powershell
python -m scripts.reproduce_paper
```

**Full retraining.** Starts from the included processed CSVs, trains target and
direct classifiers, runs all conditional diffusion jobs, creates five-seed
ensembles, aggregates results, and runs QC.

```powershell
python -m scripts.run_full_experiments --execute
```

The command generator can be audited first by omitting `--execute`.

## Final diffusion commands

The driver expands each command across horizons `0,1,2,3,4,5` and seeds
`0,1,2,3,4`.

```powershell
python -m scripts.v6_loop --region RT --fold M238 --val_holdout_mouse M229 `
  --horizons 0,1,2,3,4,5 --seeds 0,1,2,3,4 --queue anchor_b02 `
  --max_runs 0 --out_root outputs/v6_paper_RT_M238_mm_allh `
  --classifier_ckpt outputs/v6_xattn_m234_h0/classifiers/fold_M238/target/best.pth

python -m scripts.v6_loop --region RT --fold M238 --val_holdout_mouse M229 `
  --horizons 0,1,2,3,4,5 --seeds 0,1,2,3,4 --queue anchor_eeg `
  --max_runs 0 --out_root outputs/v6_paper_RT_M238_eeg_allh `
  --classifier_ckpt outputs/v6_xattn_m234_h0/classifiers/fold_M238/target/best.pth

python -m scripts.v6_loop --region VPM --fold M710 --val_holdout_mouse M1079 `
  --horizons 0,1,2,3,4,5 --seeds 0,1,2,3,4 --queue anchor_vpm `
  --max_runs 0 --out_root outputs/v6_paper_VPM_M710_allh `
  --classifier_ckpt outputs/v6_paper_VPM_classifiers/classifiers/VPM/fold_M710/target/best.pth

python -m scripts.v6_loop --region VPM --fold M710 --val_holdout_mouse M1079 `
  --horizons 0,1,2,3,4,5 --seeds 0,1,2,3,4 --queue anchor_eeg `
  --max_runs 0 --out_root outputs/v6_paper_VPM_M710_eeg_allh `
  --classifier_ckpt outputs/v6_paper_VPM_classifiers/classifiers/VPM/fold_M710/target/best.pth
```

`scripts/v6_loop.py` is the authoritative hyperparameter source for the three
anchor queues; the same values are summarized in
`configs/paper_experiments.json`.

## Expected snapshot outputs

The verification must report:

```json
{
  "status": "PASS",
  "processed_mice": 9,
  "processed_events": 1566,
  "table4_auc_cells_verified": 60,
  "paired_delta_auc_rows_verified": 24,
  "ensemble_seeds": [0, 1, 2, 3, 4],
  "samples_per_seed": 16,
  "generated_targets_per_event": 80
}
```

## Determinism scope

The checked-in paper snapshot is exactly reproducible because the held-out
probability arrays are fixed and independently re-scored. Full GPU retraining
uses fixed application seeds, but the historical training code enables cuDNN
benchmarking and does not force deterministic CUDA kernels. Therefore a fresh
training run is statistically reproducible but is not guaranteed to be
bit-for-bit identical across GPU/cuDNN versions. The code is intentionally
preserved rather than silently changing the training procedure after the
paper results were produced.

The environment recorded with the final local run was Python 3.10.19, PyTorch
2.0.1, torchvision 0.15.2, CUDA 11.8, cuDNN 8700, NumPy 1.26.4, pandas 1.5.0,
SciPy 1.12.0, scikit-learn 1.7.2, Matplotlib 3.6.0, and tqdm 1.64.1.
