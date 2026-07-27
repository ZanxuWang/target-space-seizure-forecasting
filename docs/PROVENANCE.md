# Provenance

The public snapshot was assembled on 2026-07-27 from the authors'
`final_version_for_paper` workspace, after comparing:

- the polished 20-page paper PDF;
- the final `run_final_rt_vpm_upgrade.ipynb` command driver;
- the active V6 model/training/aggregation implementation;
- all saved run configurations and training summaries;
- both final master result tables;
- all held-out classifier and ensemble probability arrays; and
- all final figure inputs and author-editable Figure 1/2 sources.

The original local Git history predates the final V6 work: run configurations
recorded commit `bda9a39`, but the V6 source and final artifacts were untracked
in that old checkout. That SHA must not be interpreted as a complete source
snapshot. The first commit of this clean repository is the authoritative
public capture of the code/artifact combination used for this release.

Absolute local source paths in processed metadata were removed. Cohort and
mouse identifiers, arrays, labels, probabilities, numerical values, and final
training scripts were not altered. Historical artifact logs may retain path
strings for traceability; these strings are metadata only and are not used by
the snapshot verifier.

The original classifier implementation and its acknowledged split convention
were preserved exactly at the authors' request.
