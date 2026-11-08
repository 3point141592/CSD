# Reproduce the results

Follow these steps to rebuild the paper's numerical summaries and figures from the released experiment records. Use Windows PowerShell and Python 3.10.11.

## 1. Prepare a working copy

Extract the full research archive and make a separate working copy. Keep the downloaded archive unchanged for comparison. The working directory must contain `requirements.txt`, `project/`, `figure_sources/` and `paper_to_files.csv`.

Open PowerShell in that directory. Keep the study folders, source files, configurations and experiment locks in their supplied layout. Use `FILE_INDEX.csv` to check the archive's file sizes and SHA-256 hashes before running analyses. Paths in this index are relative to the archive root; the index omits itself.

Use `paper_to_files.csv` to locate the source records for each table, figure and numerical claim. The GitHub repository contains selected records; use the full archive for the analysis commands below. To rebuild only the figures from a GitHub checkout, see the final section.

## 2. Create the Python environment

Select Python 3.10.11 and confirm the version:

```powershell
python --version
```

Continue when this prints `Python 3.10.11`. If another version is selected, use the path to your Python 3.10.11 executable for this check and the `-m venv` command below.

Create a new environment in the working directory, install the release requirements and check their dependencies:

```powershell
if (Test-Path -LiteralPath .\.venv) { throw 'Choose a working directory without an existing .venv.' }
python -m venv .venv
$python = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $python -m pip install --no-cache-dir -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Package installation failed; inspect the output before continuing.' }
& $python -m pip check
```

Expect `No broken requirements found.` Use `$python` for the remaining commands in this PowerShell session. The requirements pin the scientific and plotting packages, including PyTorch 2.12.1+cu132, NumPy 2.2.6 and Matplotlib 3.10.9.

## 3. Rebuild the central result summaries

Start this block at the working archive's root. It enters `project`, runs the analyses and returns to the root. Stop and inspect the output if a command reports an error.

```powershell
Push-Location .\project
& $python prepared_direct_comparison_20260921/experiment.py status
& $python prepared_direct_comparison_20260921/experiment.py analyze
& $python direct_feature_width_ablation_20261025/run.py status
& $python direct_feature_width_ablation_20261025/run.py analyze
& $python direct_feature_width_ablation_20261025/analyze_new_pair.py
Pop-Location
```

Expect 120 completed models in each study and 240 rows in each study's `per_seed.csv`: two models, two dimensions, 30 seeds and two evaluation conditions. The analysis commands validate saved records and regenerate these files under `project/`:

| Study directory | Regenerated CSV files |
| --- | --- |
| `prepared_direct_comparison_20260921` | `per_seed.csv`, `comparison_summary.csv`, `training_curves.csv`, `historical_context.csv`, `historical_summary.csv` |
| `direct_feature_width_ablation_20261025` | `per_seed.csv`, `comparison_summary.csv`, `contrasts.csv`, `new_pair_contrasts.csv`, `prior_context.csv` |

Compare the regenerated CSVs with the corresponding files in the unchanged archive. Match rows by their study identifiers, retain every seed and compare parsed numerical values at full precision. Use the rounded paper values only for the final table comparison.

## 4. Rebuild the sensitivity summaries

Run the arithmetic and validation tests:

```powershell
Push-Location .\project
& $python analysis/test_summarize_sensitivity_rerun.py
Pop-Location
```

Expect eight passing tests. Then, from the working archive's root, generate a new summary directory:

```powershell
& $python project/analysis/summarize_sensitivity_rerun.py --run project/reproduction_runs/sensitivity_rerun_20260912 --output project/reproduction_runs/sensitivity_rerun_20260912/release_summary_check
```

The output directory must not already exist and must be inside the specified run directory. For another run, choose a new output-directory name.

Expect 8,640 sensitivity-grid rows and 180 movement-cap rows to be checked. Inspect `validation.json` and compare the generated summary CSVs with the archived summaries. Use `paper_to_files.csv` to match these results to Table 9 and Appendix B.8. The script retains the filenames `table_B2_reproduction.csv` and `table_B2_reproduction.tex` for the current Table 9 results.

## 5. Rebuild and compare the figures

From the working archive's root, run:

```powershell
& $python figure_sources/Plot_Source/rebuild_figures.py
```

Expect seven PNGs and seven SVGs in `figure_sources/Figures`: `Figure1` through `Figure4` and `FigureB1` through `FigureB3`.

The regenerated PNGs retain the original figure identifiers and dimensions. Check their SHA-256 hashes against the `rebuilt_sha256` values in `figure_sources/figure_rebuild_check.json`. The publication copies are resized to 3,000 pixels wide and renumbered; their hashes intentionally differ. `figure_publication_map.csv` in the full archive, or `docs/figure_publication_map.csv` in GitHub, maps original outputs to final publication images:

```powershell
$figureRecord = Get-Content .\figure_sources\figure_rebuild_check.json -Raw | ConvertFrom-Json
foreach ($figure in $figureRecord.figures.PSObject.Properties) {
    $figurePath = Join-Path .\figure_sources\Figures $figure.Name
    $actualHash = (Get-FileHash -LiteralPath $figurePath -Algorithm SHA256).Hash
    if ($actualHash -ne $figure.Value.rebuilt_sha256) {
        throw "Figure hash mismatch: $($figure.Name)"
    }
}
Write-Output 'All seven PNG hashes match.'
```

All seven PNGs should match exactly with the specified environment. Use the PNG hashes for this identity check; SVG exports contain generated metadata.

## 6. Record the reproduction

Save the commands, console output, exit codes, regenerated summaries and comparison results. Record Python and package versions:

```powershell
& $python --version
& $python -m pip freeze > reproduced_environment.txt
```

Keep the working outputs separate from the downloaded archive. Record the archive's file-index hash alongside your results so the inputs can be identified.

These steps reproduce summaries and figures from saved records. For the central model-training protocols, consult `project/prepared_direct_comparison_20260921/PLAN.md` and `project/direct_feature_width_ablation_20261025/PLAN.md`, together with their configurations. Training fresh models and reevaluating checkpoints require their own output workspace and comparisons.

## Rebuild figures from a GitHub checkout

For a figure-only reproduction, open PowerShell at the repository root and create the environment using step 2. Then run:

```powershell
& $python Research_Archive/figure_sources/Plot_Source/rebuild_figures.py
```

The outputs appear in `Research_Archive/figure_sources/Figures`. Use `docs/figure_publication_map.csv` to match these original outputs to the resized and renumbered PNGs in `paper/`. To use the hash-check commands in step 5, first run `Push-Location .\Research_Archive`; run `Pop-Location` afterward to return to the repository root.

## Completed central reproduction

`fresh_reproduction/REPRODUCTION_REPORT.md` in the full archive records the completed 240-checkpoint replay and 240 fresh central retrains. GitHub includes the report and compact comparison evidence in `Research_Archive/fresh_reproduction/`; the full archive also includes fresh checkpoints, task arrays and execution logs. These dated records describe the manuscript and release assessed at that time. Original records and scientific outputs are preserved unchanged. The current publication assets and author-confirmed Windows provenance are described in the current manuscript.
