# Target-Space Seizure Forecasting

This is the reproducibility repository for **“From Preictal Signals to Onset-Adjacent EEG: Target-Space Seizure Forecasting with Conditional Diffusion”** It contains the exact
processed event tensors used in the paper, the active training and aggregation
code, the saved held-out probabilities behind the numerical results, and the
scripts/data required to redraw Figures 3–8.

## Reproduce the paper snapshot

Create the environment and run:

```powershell
conda env create -f environment.yml
conda activate target-space-seizure-forecasting
python -m scripts.reproduce_paper
```

This checkpoint-free workflow:

1. audits all nine processed mice and 1,566 balanced events;
2. recomputes all 60 AUC cells in Table 4 from held-out probabilities;
3. checks the five-seed, 16-samples-per-seed diffusion ensembles;
4. validates the 24 paired delta-AUC records; and
5. redraws the six scripted paper figures.

Verified tables are written to `results/paper/tables/`; recreated panels are
written to `results/paper/reproduced_figures/`. The checked-in reference panels
are in `results/paper/figures/`.

## Full training

The complete GPU command graph can be inspected without running anything:

```powershell
python -m scripts.run_full_experiments
```

Launch it with:

```powershell
python -m scripts.run_full_experiments --execute
```

Full training comprises 120 diffusion jobs (two cohorts × two conditioning
settings × six horizons × five seeds), matched direct classifiers, aggregation,
and QC. It requires a CUDA-capable GPU and substantial runtime/storage.
Checkpoints and new training outputs are written under `outputs/` and are
excluded from Git.

After full training:

```powershell
python -m scripts.v6_make_generation_figure
python -m scripts.paper_figures.make_all_paper_figures
```

## Repository layout

```text
common/                     model, data, STFT, diffusion and classifier code
configs/                    exact paper split and hyperparameter manifest
data/processed/             processed CSVs used by the paper
scripts/                    training, aggregation, QC and reproduction entry points
scripts/paper_figures/      final-paper Figure 3–8 renderers
results/paper/artifacts/    compact held-out probabilities/configs/logs (no weights)
results/paper/figure_data/  cached arrays required for diagnostic panels
results/paper/figures/      checked-in final figure files
docs/                       code mapping, commands, provenance and audit notes
```


