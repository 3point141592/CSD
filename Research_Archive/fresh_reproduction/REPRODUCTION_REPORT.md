# CSD central experiment reproduction

Generated 2026-10-31 22:38 UTC from the completed run records.

**PASS: the saved-checkpoint replay, fixed pilot, and all 240 central retrains passed the predeclared checks.**

This run evaluates the 240 archived central-model checkpoints, then trains 240 new models from initialization: prepared-plus-wide Direct, Relational, prepared-plus-narrow Direct and raw-plus-wide Direct, each at 32D and 64D across the original 30 seeds. The fixed pilot used seeds 7, 19 and 43, producing 24 of those 240 fresh models. The remaining 216 models were added after the pilot passed; the pilot models were not counted twice.

| Verification | Result |
|---|---|
| saved checkpoint replay | Pass |
| fixed pilot gate | Pass |
| full gate | Pass |
| same predeclared policy | Pass |
| all 240 fresh completion timestamps | Pass |
| all 240 checkpoint states exact | Pass |
| all 240 training curves exact | Pass |
| all 16 groups and 4 contrasts exact | Pass |
| fresh empty start documented | Pass |
| fresh training execution recorded | Pass |
| all six regenerated result files exact | Pass |
| original archive kit and copied inputs unchanged | Pass |

The saved-checkpoint replay produced 480 metric rows and 14,760 numeric comparisons: 14,760 were exact and 0 exceeded the predeclared tolerances. The full fresh-training comparison checked 14,760 numeric values: 14,760 were exact and 0 exceeded tolerance. All exactness counts are read from the detailed phase reports, not inferred from rounded paper values.

The full comparison records 240 exact checkpoint state dictionaries and 240 exact training histories out of 240. Frozen source/configuration comparisons checked 112 files, with 0 mismatches. All 240 completion timestamps were independently inspected for this report against the predeclared fresh-run boundary `2026-10-31T21:42:56.791285+00:00`; 0 predate it. Fresh completions span `2026-10-31T21:46:01+00:00` to `2026-10-31T22:36:42.015136+00:00`.

All six regenerated result CSVs match their archived scientific fields exactly, covering 16,868 numeric field comparisons. These overlap with the per-model checks and are not additional independent experiments.

The comparison policy was fixed before the pilot. Configurations, seeded-data fingerprints, source identities, checkpoint tensors and training histories were required to match exactly for automatic continuation. Fraction metrics allowed at most 1e-06 absolute numerical difference, other continuous metrics at most 1e-05, with zero relative tolerance. Timing and memory measurements were recorded separately and were not pass/fail criteria. The 48 performance-endpoint rows in the prepared summary were excluded from scientific summary equality checks.

## Central coverage results

Coverage values below are percentages, shown as mean ± sample SD across 30 seeds. Differences are percentage points. The accompanying `Central_results_comparison.csv` retains full-precision fractions, useful-candidate and decoy statistics, and the four paired contrasts.

| Dimension | Model | Condition | Archived mean ± SD (%) | Fresh mean ± SD (%) | Fresh minus archived (pp) |
|---|---|---|---|---|---|
| 32 | Prepared + wide Direct | Exact rotation | 63.518524 ± 5.613365 | 63.518524 ± 5.613365 | +0.000000000 |
| 32 | Prepared + wide Direct | IID | 99.777778 ± 1.217160 | 99.777778 ± 1.217160 | +0.000000000 |
| 32 | Relational | Exact rotation | 99.666668 ± 1.058121 | 99.666668 ± 1.058121 | +0.000000000 |
| 32 | Relational | IID | 99.666668 ± 1.058121 | 99.666668 ± 1.058121 | +0.000000000 |
| 32 | Prepared + narrow Direct | Exact rotation | 0.185185 ± 0.512370 | 0.185185 ± 0.512370 | +0.000000000 |
| 32 | Prepared + narrow Direct | IID | 84.481487 ± 9.150756 | 84.481487 ± 9.150756 | +0.000000000 |
| 32 | Raw + wide Direct | Exact rotation | 0.185185 ± 0.421166 | 0.185185 ± 0.421166 | +0.000000000 |
| 32 | Raw + wide Direct | IID | 97.777781 ± 3.690907 | 97.777781 ± 3.690907 | +0.000000000 |
| 64 | Prepared + wide Direct | Exact rotation | 63.148154 ± 4.435656 | 63.148154 ± 4.435656 | +0.000000000 |
| 64 | Prepared + wide Direct | IID | 99.851852 ± 0.811439 | 99.851852 ± 0.811439 | +0.000000000 |
| 64 | Relational | Exact rotation | 99.777779 ± 0.538035 | 99.777779 ± 0.538035 | +0.000000000 |
| 64 | Relational | IID | 99.777779 ± 0.538035 | 99.777779 ± 0.538035 | +0.000000000 |
| 64 | Prepared + narrow Direct | Exact rotation | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | +0.000000000 |
| 64 | Prepared + narrow Direct | IID | 2.518519 ± 3.451708 | 2.518519 ± 3.451708 | +0.000000000 |
| 64 | Raw + wide Direct | Exact rotation | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | +0.000000000 |
| 64 | Raw + wide Direct | IID | 17.740742 ± 9.762979 | 17.740742 ± 9.762979 | +0.000000000 |

## Prepared Direct minus Relational paired coverage contrasts

Values are percentage points with 95% paired-bootstrap intervals. These are the original 20,000-resample analysis outputs, recomputed by the unchanged analysis program after fresh training.

| Dimension | Condition | Archived difference [95% CI] | Fresh difference [95% CI] |
|---|---|---|---|
| 32 | Exact rotation | -36.148144 [-38.111107, -34.148143] | -36.148144 [-38.111107, -34.148143] |
| 32 | IID | +0.111110 [-0.518519, +0.666665] | +0.111110 [-0.518519, +0.666665] |
| 64 | Exact rotation | -36.629625 [-38.148145, -35.111106] | -36.629625 [-38.148145, -35.111106] |
| 64 | IID | +0.074073 [-0.333333, +0.370369] | +0.074073 [-0.333333, +0.370369] |

## Execution time

| Study | Fresh models | Archived optimizer total | Fresh optimizer total | Fresh summed job wall time |
|---|---|---|---|---|
| Prepared comparison | 120 | 21.29 min | 19.75 min | 20.12 min |
| Added feature/width controls | 120 | 24.95 min | 20.48 min | 20.78 min |

Total fresh optimizer time was 40.22 minutes (2413.39 seconds). Summed per-job wall time was 40.90 minutes (2454.05 seconds); it includes each job's immediate evaluation but does not capture every setup, data-preparation or comparison cost. Pilot plus full-phase elapsed time was 51.06 minutes (3063.83 seconds). Including the saved-checkpoint replay, the summed phase elapsed time was 55.33 minutes (3320.05 seconds). The clock interval from pilot start to full completion, including gaps between phases, was 51.30 minutes (3078.00 seconds). Timings are descriptive and do not determine numerical success.

## Environment and interpretation

The run used `3.10.11 (main, May  7 2023, 18:05:05) [MSC v.1929 64 bit (AMD64)]`, PyTorch `2.12.1+cu132`, NumPy `2.2.6`, CUDA `13.2`, GPU `NVIDIA GeForce RTX 5090` and driver `610.60`. Dtype was `float32`; TF32 matmul/cuDNN flags were `False`/`False`. The archived deterministic-algorithm setting was preserved as `False`. Runtime identity was checked before training and is retained in the phase execution records.

This is fresh same-environment reproduction of the central prepared/feature-width comparisons. It does not retrain every earlier matched, wider-head, symmetry, selector, corruption or historical study in the paper. It does not establish independent external replication or performance on new task families. The historical CPU sensitivity execution-environment provenance remains unresolved; this new GPU run provides a documented central reproduction and does not reconstruct that missing history.

## Original-file integrity

The final hash audit verified all 7,915 indexed source-archive files, all 352 indexed revised-kit files, and all 295 copied input pairs against their recorded identities. No fixed input changed, and no unindexed files appeared in the archive or kit. The kit's assessed manuscript SHA256 remains `2bd4096e27453e52c7ab70f1ef7002886cdd309fcbdd3f3ebba1c4debb7ad9a0`.

The separately located live author DOCX could not be re-verified (PermissionError). The assessed manuscript copy in the revised kit does match the recorded author hash; the fresh results apply to that version.

## Evidence

- `checkpoint_replay/summary.json` and `metric_comparisons.csv`: archived-model re-evaluation.
- `pilot_comparison/summary.json`: fixed 24-model continuation gate.
- `full_comparison/summary.json`, `per_seed_metric_comparison.csv`, `checkpoint_comparisons.json`, `curve_comparisons.json` and `regenerated_summary_comparisons.json`: complete fresh-model comparisons.
- `pilot_execution.json`, `full_execution.json`, `pilot_result.json` and `full_result.json`: commands, environment and elapsed records.
- `comparison_policy.json`: predeclared criteria, SHA256 `af41b60c6cb416add08db9085a5bdebe7ed41a00c2440f63fbd81280fffa2a98`.
- `project/prepared_direct_comparison_20260921/jobs` and `project/direct_feature_width_ablation_20261025/jobs`: fresh checkpoints and per-job artifacts.
- `setup_manifest.json`: copied frozen-source identities and empty starting job directories.
- `integrity_final.json`: final archive, kit, copied-input and manuscript hash audit, including the live-document access limitation.

Original reference archive: `C:\Users\krivi\OneDrive\Desktop\Creativity\CSD_PeerJ_Evidence_Package_20261028_v1\02_Research_Archive`. Fresh-run folder: `C:\Users\krivi\Documents\Codex\2026-10-30\ins\outputs\CSD_Fresh_Reproduction_2026-10-31`. The reporting program reads these records and writes only this report and the central-comparison CSV; it does not alter the manuscript, kit, source archive or scientific results.
