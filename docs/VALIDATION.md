# Release validation

The public snapshot was validated with the recorded Python 3.10/PyTorch 2.0.1
environment on 2026-07-27.

| Check | Result |
|---|---|
| Python compilation of `common/` and `scripts/` | PASS |
| Processed mice | 9 |
| Processed events | 1,566 |
| Signal shape per event/modality | 5,376 samples |
| STFT/dataset smoke test | past `(3,31,27)`, target `(31,27)` |
| Table 4 AUC cells recomputed | 60/60 PASS |
| Paired delta-AUC rows cross-checked | 24/24 PASS |
| Diffusion ensemble metadata | 5 seeds × 16 = 80 targets/event |
| Full command graph dry run | PASS |
| Figures 3–8 regenerated | PASS |
| Regenerated/reference PNG SHA-256 equality | 6/6 exact |

The exact per-file hashes for all source, processed data, compact artifacts,
paper figures, editable figure sources, and the paper PDF are recorded in
`MANIFEST.sha256`.

The one-command validation entry point is:

```powershell
python -m scripts.reproduce_paper
```
