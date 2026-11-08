The original evaluator, matching function, task quality, region rule and refiner are frozen in source_snapshot. Added diagnostics reuse their outputs or exact definitions.

| Reported quantity | Definition |
|---|---|
| IID useful-region coverage (`mode_set_recall`) | After refinement, distinct useful regions reached by at least one candidate meeting quality >=0.62, divided by the three useful regions; averaged over the original30 test starts. |
| Exact-rotation coverage | The identical rule on the original per-start CUDA rotation stream, seed+81001, with both task geometry and starts transformed. |
| Initial matched directional cosine (`mean_matched_cosine`) | Mean cosine to useful-center directions before refinement, using the original exact assignment minimizing `1-cosine + 0.1*(alpha-target_distance)^2`. This is the joint loss's assignment, not a separately optimized cosine-only assignment. |
| Any-useful success (`proposal_recall`) | Fraction of test starts with at least one useful candidate after refinement. |
| Useful-candidate fraction | Fraction of all eight candidates satisfying the same useful-region and quality>=0.62 mask after refinement. |
| Decoy fraction (`candidate_decoy_rate`) | Fraction whose original region index exceeds the number of useful regions. The original rule does not additionally apply the useful-quality threshold to decoys. |
| Training fit | Final supervised loss and original matching metrics over all7680 training records. Diagnostic evaluation uses eight chunks of960; optimization is unchanged at batch960. |
| Raw/directional output rank | Numerical SVD rank of all30*8 raw or normalized proposal vectors stacked into a matrix with d columns. Threshold is `max(matrix.shape)*float32_epsilon*largest_singular_value`; spectra are saved. It describes these evaluated outputs and this tolerance, not model quality or a universal capacity bound. |
| Decoder augmented rank | For Direct, numerical rank of `[W2,bias]`, including the bias column. No width16 cap is assumed. |
| Parameter count | Sum of trainable parameter elements, excluding buffers such as candidate codes and normalization statistics. |
| Optimizer seconds | Original trainer's synchronized optimizer-loop wall time. |
| Job/pair wall time | Job time includes training, final training fit, evaluation and checkpoint verification; pair time also includes shared data generation, feature construction, data serialization, checks, historical replay and commits. Pilot time includes both dimension pairs and GPU historical replay. |
| GPU memory | PyTorch peak allocated/reserved bytes, with a monitor retaining peaks across data preparation and trainer counter resets. This is framework allocation, not total device memory used by every application. |
| Host memory | Peak process resident memory sampled during each pair. |

The proposal scale is capped at6 before refinement. The unchanged refiner then performs nine steps of size0.28 with gradient norm capped at4; the proposal cap is not a new cap on the total refined displacement.

The primary replicate is the independently trained seed. Each dimension has30 paired seeds, with common tasks, starts, component permutations and candidate-code seeds between models. Starts, candidates and the three goal conditions within a seed are not independent training replicates. `sample_sd` uses denominator n-1. `ci95_low/high` are percentile bootstrap confidence limits from20000 resamples of paired seed indices; they are distinct from SD and are not adjusted for multiple secondary endpoints.

All current model inputs are supplied by the benchmark: current state, fixed candidate codes, learned task token, and unlabeled component geometry. Useful-center membership, target directions and target distances remain in the supervised matching/loss or evaluation paths. Historical scores and saved-model CPU replay are labeled separately from the primary contemporaneous GPU comparison.
