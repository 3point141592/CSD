# Code map

Paths below are inside `Research_Archive/project`.

| Purpose | Source |
|---|---|
| Task generation and local refinement | `creativity_toy/landscapes_d16.py`, `gen3_gpu/task_batch.py` |
| Direct representation and decoder | `gen3_gpu/landscape_representation.py` |
| Relational vectors and transformations | `gen3_gpu/symmetry.py` |
| Supervised objective and metrics | `gen3_gpu/training_landscape_supervised.py` |
| Matched dimension comparison | `paper_csd_toy_revision5/scripts/run_matched_scaling_study.py` |
| Prepared comparison | `prepared_direct_comparison_20260921/experiment.py`, `analyze.py`, and `source_snapshot/` |
| Added feature/width cells | `direct_feature_width_ablation_20261025/run.py`, `analyze_new_pair.py` |
| Head-only intervention | `analysis/run_direct_head_ablation.py`, `resume_direct_head_ablation.py` |
| Coordinate shifts | `gen3_gpu/study_symmetry.py` |
| Grouped selectors | `analysis/run_fair_selector_verification.py` |
| Rank and multiplicity | `analysis/direct_rank_diagnostic.py`, `multiplicity_diagnostic.py` |
| Reference corruption and cone | `paper_csd_toy_revision5/scripts/run_reference_corruption.py`, `analysis/run_corruption_reachability_audit.py` |
| CPU refinement sensitivity | `analysis/run_sensitivity_rerun.py`, `summarize_sensitivity_rerun.py` |

The prepared comparison loads the code in its own `source_snapshot` directory before the top-level packages. The feature-width ablation imports the prepared comparison runner, so the supplied directory structure is required when running these analyses.

The hashes in `docs/file_hashes.csv` identify the files supplied in this repository. The full research archive provides the original frozen source used for the experiments.