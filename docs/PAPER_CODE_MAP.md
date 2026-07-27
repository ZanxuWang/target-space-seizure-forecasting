# Paper-to-code and artifact map

This map was constructed from the polished 20-page paper, the final driver
notebook, saved run configurations/logs, master result tables, probability
arrays, figure source arrays, and the active V6 implementation.

## Implementation

| Paper element | Active source | Evidence/output |
|---|---|---|
| Event windows, cohort mice, six horizons | `common/config.py` | `configs/paper_experiments.json`, `data/processed/summary.csv` |
| CSV loading and 21 s event tensors | `common/datasets.py`, `common/v6_datasets.py` | `data/processed/{RT,VPM}/{mouse}/` |
| Hann STFT, nFFT 128, hop 32, ≤60 Hz, 31×27 | `common/spectrogram.py` | tensors consumed by all classifiers/diffusion jobs |
| EEG target/direct classifier | `common/classifier.py`, `scripts/train_classifier.py` | classifier summaries and `final_eval_probs.npz` |
| Full-MM direct classifier | `common/classifier_mm.py`, `scripts/train_classifier_mm.py` | `classifiers_mm/.../final_eval_probs.npz` |
| Cosine diffusion schedule | `common/diffusion.py` | V6 checkpoint configuration |
| Cross-attention U-Net, spatial target tokens, past encoder, CFG, EMA, DDIM | `common/v6_diffusion.py` | per-seed configs/logs under `results/paper/artifacts/` |
| SpecAugment, cross-mouse mixup, auxiliary losses, GRL mouse adversary | `common/v6_datasets.py`, `scripts/v6_train.py` | `configs/paper_experiments.json`; per-run `config.json` |
| 6 horizons × 5 seeds orchestration | `scripts/v6_loop.py` queues `anchor_b02`, `anchor_vpm`, `anchor_eeg` | 120 per-seed final-evaluation artifacts |
| Five-seed/80-target ensemble and bootstrap CIs | `scripts/v6_aggregate_results.py` | both `master_results.tsv` files and ensemble NPZs |
| Quality-control audit | `scripts/v6_qc_audit.py` | compact audit/config/log evidence |

## Numerical results

| Paper result | Reproduction source | Reproduction command |
|---|---|---|
| Event counts and class balance | processed `labels.csv` files | `python -m scripts.verify_paper_results` |
| Table 4 target, direct and diffusion AUCs | saved held-out probability NPZs | `python -m scripts.verify_paper_results` |
| RT/M238 master results | `results/paper/artifacts/v6_paper_RT_M238_mm_allh/master_results.tsv` | `python -m scripts.v6_aggregate_results --version v6_paper_RT_M238_mm_allh ...` |
| TC/M710 master results | `results/paper/artifacts/v6_paper_VPM_M710_allh/master_results.tsv` | `python -m scripts.v6_aggregate_results --version v6_paper_VPM_M710_allh --region VPM ...` |
| 24 paired delta-AUC tests | `results/paper/figure_data/figure8_delta_auc_statistics.tsv` (historical filename) | `python -m scripts.paper_figures.figure3_delta_auc_statistics` |

## Figures

Final paper numbering differs from the historical local output filenames. The
public scripts use the final numbering.

| Paper figure | Public source | Inputs | Output stem |
|---|---|---|---|
| Figure 1, conceptual pipeline | editable author source | `assets/editable_figures/Figure1_editable.pptx` | checked-in Figure 1 PNG |
| Figure 2, architecture | editable author source | `assets/editable_figures/Figure2_editable.pptx` | checked-in Figure 2 PNG |
| Figure 3, paired delta AUC | `scripts/paper_figures/figure3_delta_auc_statistics.py` | held-out direct/diffusion probabilities | `figure3_delta_auc_statistics` |
| Figures 4–5, RT/TC ROC grids | `scripts/paper_figures/figure4_5_roc_performance.py` | held-out probabilities and target-alignment arrays | `figure4_*`, `figure5_*` |
| Figures 6–7, generation quality | `scripts/paper_figures/figure6_7_generation_quality.py` | cached generation prototype arrays | `figure6_*`, `figure7_*` |
| Figure 8, target-space diagnostics | `scripts/paper_figures/figure8_target_space_diagnostic.py` | cached target-space alignment arrays | `figure8_*` |

Run every scripted panel with:

```powershell
python -m scripts.paper_figures.make_all_paper_figures
```

The cached diagnostic arrays were created from the five selected horizon-0
checkpoints by `scripts/v6_make_generation_figure.py` using 50 DDIM steps,
guidance scale 2, and seed 2026.

## Exact final experiment identities

| Cohort/method | Version/root | Fold | Validation mouse |
|---|---|---|---|
| RT full-MM diffusion | `v6_paper_RT_M238_mm_allh` | M238 | M229 |
| RT EEG diffusion | `v6_paper_RT_M238_eeg_allh` | M238 | M229 |
| RT EEG classifier | `v6_paper_RT_M238_eeg_classifiers_valauc` | M238 | implementation-preserved internal split |
| RT full-MM classifier | `v6_rt_h0_fold_screen` | M238 | implementation-preserved internal split |
| TC full-MM diffusion | `v6_paper_VPM_M710_allh` | M710 | M1079 |
| TC EEG diffusion | `v6_paper_VPM_M710_eeg_allh` | M710 | M1079 |
| TC EEG classifier | `v6_paper_VPM_M710_eeg_classifiers_valauc` | M710 | implementation-preserved internal split |
| TC full-MM classifier | `v6_paper_VPM_classifiers` | M710 | implementation-preserved internal split |
