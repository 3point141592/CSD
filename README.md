# CSD

Code and saved results for **Geometry and supplied information in learned proposals for local search**.

The study compares coordinate predictions with weighted reference-vector predictions for local search on synthetic Gaussian landscapes. It examines model capacity, useful-region coverage, coordinate changes and the information supplied by the benchmark.

## Read and reproduce

- `paper/`: the manuscript, seven figures (1–7), ten tables (1–10) and data supplement.
- `Research_Archive/project/`: scientific code and source snapshots, with selected configurations/results.
- `Research_Archive/figure_sources/`: code and inputs for the seven figures.
- `Research_Archive/paper_to_files.csv`: sources for the paper's tables, figures and main numerical claims.
- [Reproduction guide](docs/reproduce.md): environments, saved-result checks and expected outputs.
- [Code map](docs/code_map.md): where the main calculations are implemented.

 The [reproduction guide](docs/reproduce.md) distinguishes workflows that use this repository from those that require the full archive. Repository file sizes and SHA-256 hashes are recorded in `docs/file_hashes.csv`. The full archive preserves the source-code versions used for the original experiments.

## Status

This release accompanies the manuscript prepared for PeerJ Computer Science. Publication assets use the final journal numbering; original scientific source filenames remain unchanged. Verification reconstructed the saved numerical results and regenerated all seven figures. A separate reproduction evaluated 240 archived central-model checkpoints and trained 240 new central models from initialization; the compared scientific metrics, checkpoint tensors and training histories matched exactly. The [reproduction report](Research_Archive/fresh_reproduction/REPRODUCTION_REPORT.md) and comparison records provide the evidence. The reproduction guide describes the tested environment and distinguishes saved-record reconstruction from fresh training.

## License

The [MIT License](LICENSE) applies to the original code, accompanying software
documentation and original generated research files included in this release,
including synthetic datasets, trained model checkpoints, numerical results
and figure images, to the extent the author holds rights in them.
Copyright (c) 2026 Krivan Mirok. Original code in frozen source snapshots is
included. Third-party materials remain subject to their own licenses.

The manuscript's publication license is handled separately; this MIT statement
does not assign a publication license to the manuscript.

## Repository and archive

Repository: https://github.com/3point141592/CSD

Full research archive: https://doi.org/10.5281/zenodo.24047087. `docs/publication_numbering.csv` maps original table and figure identifiers to the final paper; `docs/figure_publication_map.csv` records source and publication image hashes.
