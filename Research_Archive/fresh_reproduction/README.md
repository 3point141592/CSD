# CSD fresh central experiment reproduction

Completed 31 October 2026. **PASS:** 240 saved checkpoints were evaluated again, and 240 new central models were trained from initialization. All compared scientific metrics, checkpoint tensors and training histories match the archive exactly.

Start with [REPRODUCTION_REPORT.md](REPRODUCTION_REPORT.md). [Central_results_comparison.csv](Central_results_comparison.csv) contains the 16 coverage groups and four prepared paired contrasts at full precision.

The training set comprises four configurations, two dimensions (32 and 64), and the original 30 seeds. The fixed 24-model pilot is included once in the 240-model total. All models received 1,500 optimizer updates. The run used the original research environment and RTX 5090.

This confirms same-environment reproduction of the central comparisons. Earlier historical experiments were not all retrained, and the historical CPU sensitivity provenance remains unresolved. The assessed manuscript is the byte-preserved copy in the revised kit; the separate live OneDrive Word document could not be reopened. See the report for these limits and the final integrity audit.

The original archive and revised kit are unchanged. This folder retains the new evidence separately. Copied study documents describe the original experiments; use the reproduction report above for the outcome of this new run.

Compact directory tree:

```text
CSD_Fresh_Reproduction_2026-10-31/
|-- README.md
|-- REPRODUCTION_REPORT.md
|-- Central_results_comparison.csv
|-- comparison_policy.json
|-- setup_manifest.json
|-- integrity_final.json
|-- run_status.json
|-- pilot_execution.json / pilot_result.json
|-- full_execution.json / full_result.json
|-- checkpoint_replay/          Re-evaluation records for 240 archived models
|-- pilot_comparison/           Fixed 24-model continuation checks
|-- full_comparison/            All 240 fresh-model and summary comparisons
|-- execution_logs/             Replay, pilot and full-run console logs
|-- validation_scripts/         Separate orchestration, comparison and reporting code
`-- project/
    |-- prepared_direct_comparison_20260921/
    |   |-- jobs/               120 newly trained models and their run records
    |   `-- data/               60 regenerated seeded datasets
    |-- direct_feature_width_ablation_20261025/
    |   `-- jobs/               120 newly trained feature/width controls
    |-- paper_csd_toy_revision5/               Copied historical references
    `-- Paper_B_Publication_Materials_20260912/ Copied historical references
```

`setup_manifest.json` identifies every copied input. New checkpoints are in the two central studies' `jobs` directories. `comparison_policy.json` records the criteria fixed before training. Timing and memory measurements were retained separately from scientific equality checks.
