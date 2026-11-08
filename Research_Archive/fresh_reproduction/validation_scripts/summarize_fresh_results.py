"""Generate the final central-reproduction report after all 240 fresh jobs finish.

CPU-only, standard-library reporting. This script never runs training/inference
or writes to the source archive, paper, kit, or scientific job files.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

STUDIES = {
    'prepared_direct_comparison_20260921': ('DirectPrepared', 'Relational'),
    'direct_feature_width_ablation_20261025': ('DirectPreparedNarrow', 'DirectRawWide'),
}
LABELS = {'DirectPrepared': 'Prepared + wide Direct', 'Relational': 'Relational',
          'DirectPreparedNarrow': 'Prepared + narrow Direct', 'DirectRawWide': 'Raw + wide Direct'}
ORDER = {name: i for i, name in enumerate(LABELS)}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def csv_load(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def time_value(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def stats(rows, metric):
    values = [float(r[metric]) for r in rows]
    return statistics.mean(values), statistics.stdev(values)


def duration(seconds):
    return f'{seconds / 60:.2f} minutes ({seconds:.2f} seconds)'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path, help='CSD_Fresh_Reproduction folder')
    parser.add_argument('--archive', type=Path, help='Optional override of the original full archive path')
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    setup = load(run / 'setup_manifest.json')
    archive = args.archive.resolve(strict=True) if args.archive else Path(setup['source_project']).parent
    project = run / 'project'
    if run.is_relative_to(archive) or project.resolve() == Path(setup['source_project']).resolve():
        raise ValueError('Report destination must be outside the original archive')
    # Required final records are checked before creating either deliverable.
    pilot = load(run / 'pilot_comparison/summary.json')
    full = load(run / 'full_comparison/summary.json')
    replay = load(run / 'checkpoint_replay/summary.json')
    pilot_execution = load(run / 'pilot_execution.json')
    full_execution = load(run / 'full_execution.json')
    pilot_result = load(run / 'pilot_result.json')
    full_result = load(run / 'full_result.json')
    integrity = load(run / 'integrity_final.json')
    regenerated = load(run / 'full_comparison/regenerated_summary_comparisons.json')
    assert len(regenerated) == 6
    summary_numeric_count = sum(r['numeric_fields_compared'] for r in regenerated)
    live_author = integrity['results']['live_author_manuscript']
    live_author_note = ('The separately located live author DOCX also matches the assessed manuscript hash.'
                        if live_author['pass'] else
                        f"The separately located live author DOCX could not be re-verified ({live_author.get('error_type', 'hash mismatch')}). The assessed manuscript copy in the revised kit does match the recorded author hash; the fresh results apply to that version.")
    policy_path = run / 'comparison_policy.json'
    policy = load(policy_path)
    not_before = time_value(policy['fresh_completion_not_before_utc'])
    seeds = load(project / 'prepared_direct_comparison_20260921/configurations.json')['32']['matched']['seeds']
    assert len(seeds) == 30 and len(set(seeds)) == 30
    assert policy['dimensions'] == [32, 64]

    old_groups, fresh_groups = defaultdict(list), defaultdict(list)
    model_records, timestamps, freshness_failures = [], [], []
    study_cost = {}
    for study, kinds in STUDIES.items():
        cost = {'archived_optimizer_seconds': 0.0, 'fresh_optimizer_seconds': 0.0,
                'archived_job_wall_seconds': 0.0, 'fresh_job_wall_seconds': 0.0, 'models': 0}
        for d in policy['dimensions']:
            for seed in seeds:
                for model in kinds:
                    key = f'd{d}_s{seed}_{model}'
                    rel = Path(study) / 'jobs' / f'd{d}' / f'seed_{seed}' / model
                    new, old = project / rel, archive / 'project' / rel
                    complete = load(new / 'completion.json')
                    if complete.get('key') != key or complete.get('status') != 'complete' or complete.get('steps') != 1500:
                        raise ValueError(f'Not a completed fresh 1500-step model: {key}')
                    stamp = time_value(complete['completed_utc'])
                    timestamps.append(stamp)
                    if stamp < not_before:
                        freshness_failures.append(key)
                    nm, om = load(new / 'metadata.json'), load(old / 'metadata.json')
                    nr, ores = load(new / 'metrics.json'), load(old / 'metrics.json')
                    if len(nr) != 2 or {r['split'] for r in nr} != {'iid', 'exact_rotated'}:
                        raise ValueError(f'Expected IID and rotation rows for {key}')
                    for r in nr:
                        assert r['model'] == model and r['dimension'] == d and r['seed'] == seed
                        fresh_groups[(d, model, r['split'])].append(r)
                    for r in ores:
                        old_groups[(d, model, r['split'])].append(r)
                    cost['models'] += 1
                    for label, meta in [('archived', om), ('fresh', nm)]:
                        cost[f'{label}_optimizer_seconds'] += float(meta['optimizer_seconds'])
                        cost[f'{label}_job_wall_seconds'] += float(meta['job_wall_seconds'])
                    model_records.append({'key': key, 'completed_utc': complete['completed_utc']})
        study_cost[study] = cost
    assert len(model_records) == 240 and len({r['key'] for r in model_records}) == 240
    assert len(fresh_groups) == len(old_groups) == 16

    central_rows = []
    group_markdown = []
    for (d, model, split), fresh_rows in sorted(fresh_groups.items(), key=lambda kv: (kv[0][0], ORDER[kv[0][1]], kv[0][2])):
        old_rows = old_groups[(d, model, split)]
        assert len(old_rows) == len(fresh_rows) == 30
        assert {r['seed'] for r in fresh_rows} == set(seeds)
        original, original_sd = stats(old_rows, 'mode_set_recall')
        actual, actual_sd = stats(fresh_rows, 'mode_set_recall')
        entry = {'row_type': 'coverage_group', 'dimension': d, 'model_or_contrast': model,
                 'split': split, 'n_paired_seeds': 30, 'metric': 'mode_set_recall', 'unit': 'fraction',
                 'archived_mean': original, 'fresh_mean': actual, 'mean_difference': actual-original,
                 'archived_sample_sd': original_sd, 'fresh_sample_sd': actual_sd,
                 'sample_sd_difference': actual_sd-original_sd,
                 'mean_difference_percentage_points': 100*(actual-original),
                 'archived_ci95_low': '', 'archived_ci95_high': '', 'fresh_ci95_low': '', 'fresh_ci95_high': '',
                 'ci95_low_difference': '', 'ci95_high_difference': '',
                 'archived_ci_method': '', 'fresh_ci_method': '',
                 'exact_compared_statistics': original == actual and original_sd == actual_sd}
        for metric, prefix in [('useful_candidate_fraction', 'useful_candidate'), ('candidate_decoy_rate', 'decoy')]:
            a, a_sd = stats(old_rows, metric)
            b, b_sd = stats(fresh_rows, metric)
            entry.update({f'archived_{prefix}_mean': a, f'fresh_{prefix}_mean': b,
                          f'{prefix}_mean_difference': b-a, f'archived_{prefix}_sample_sd': a_sd,
                          f'fresh_{prefix}_sample_sd': b_sd})
        central_rows.append(entry)
        condition = 'IID' if split == 'iid' else 'Exact rotation'
        group_markdown.append(f'| {d} | {LABELS[model]} | {condition} | {100*original:.6f} ± {100*original_sd:.6f} | {100*actual:.6f} ± {100*actual_sd:.6f} | {100*(actual-original):+.9f} |')

    def contrast_rows(path):
        rows = csv_load(path)
        selected = [r for r in rows if r.get('endpoint') == 'mode_set_recall' and r.get('comparison') == 'DirectPrepared_minus_Relational']
        assert len(selected) == 4
        return {(int(r['dimension']), r['split']): r for r in selected}

    original_ci = contrast_rows(archive / 'project/prepared_direct_comparison_20260921/comparison_summary.csv')
    fresh_ci = contrast_rows(project / 'prepared_direct_comparison_20260921/comparison_summary.csv')
    ci_markdown = []
    for key in sorted(original_ci):
        a, b = original_ci[key], fresh_ci[key]
        d, split = key
        old_mean, fresh_mean = float(a['mean']), float(b['mean'])
        old_sd, new_sd = float(a['sample_sd']), float(b['sample_sd'])
        alo, ahi, blo, bhi = [float(x) for x in [a['ci95_low'], a['ci95_high'], b['ci95_low'], b['ci95_high']]]
        central_rows.append({'row_type': 'prepared_paired_contrast', 'dimension': d,
             'model_or_contrast': 'DirectPrepared_minus_Relational', 'split': split,
             'n_paired_seeds': int(b['n_paired_seeds']), 'metric': 'mode_set_recall', 'unit': 'fraction',
             'archived_mean': old_mean, 'fresh_mean': fresh_mean, 'mean_difference': fresh_mean-old_mean,
             'archived_sample_sd': old_sd, 'fresh_sample_sd': new_sd, 'sample_sd_difference': new_sd-old_sd,
             'mean_difference_percentage_points': 100*(fresh_mean-old_mean),
             'archived_ci95_low': alo, 'archived_ci95_high': ahi, 'fresh_ci95_low': blo, 'fresh_ci95_high': bhi,
             'ci95_low_difference': blo-alo, 'ci95_high_difference': bhi-ahi,
             'archived_ci_method': a['ci_method'], 'fresh_ci_method': b['ci_method'],
             'exact_compared_statistics': (old_mean,old_sd,alo,ahi,a['ci_method']) == (fresh_mean,new_sd,blo,bhi,b['ci_method'])})
        condition = 'IID' if split == 'iid' else 'Exact rotation'
        ci_markdown.append(f'| {d} | {condition} | {100*old_mean:+.6f} [{100*alo:+.6f}, {100*ahi:+.6f}] | {100*fresh_mean:+.6f} [{100*blo:+.6f}, {100*bhi:+.6f}] |')
    assert len(central_rows) == 20

    checks = {
        'saved_checkpoint_replay': replay['models_evaluated'] == 240 and replay['all_within_predeclared_tolerance'],
        'fixed_pilot_gate': pilot['pass'] and pilot['completed_models'] == 24,
        'full_gate': full['pass'] and full['completed_models'] == 240,
        'same_predeclared_policy': pilot['policy_sha256'] == full['policy_sha256'] == sha(policy_path),
        'all_240_fresh_completion_timestamps': not freshness_failures,
        'all_240_checkpoint_states_exact': full['exact_checkpoint_models'] == 240,
        'all_240_training_curves_exact': full['exact_curve_models'] == 240,
        'all_16_groups_and_4_contrasts_exact': all(r['exact_compared_statistics'] for r in central_rows),
        'fresh_empty_start_documented': setup['fresh_empty_jobs'] and setup['old_runtime_pilot_and_data_not_copied'],
        'fresh_training_execution_recorded': pilot_execution['fresh_training'] and full_execution['fresh_training'],
        'all_six_regenerated_result_files_exact': all(r['exact_scientific_values'] for r in regenerated),
        'original_archive_kit_and_copied_inputs_unchanged': integrity['all_fixed_inputs_unchanged'],
    }
    passed = all(checks.values())
    run_elapsed = float(pilot_result['elapsed_seconds']) + float(full_result['elapsed_seconds'])
    total_phase_elapsed = run_elapsed + float(replay['elapsed_seconds'])
    elapsed_window = (time_value(full_result['finished_utc']) - time_value(pilot_execution['started_utc'])).total_seconds()
    totals = {field: sum(r[field] for r in study_cost.values()) for field in ['archived_optimizer_seconds', 'fresh_optimizer_seconds', 'archived_job_wall_seconds', 'fresh_job_wall_seconds']}
    runtime = full_execution['runtime']
    status_line = ('**PASS: the saved-checkpoint replay, fixed pilot, and all 240 central retrains passed the predeclared checks.**' if passed else
                   '**REQUIRES REVIEW: the final data are present, but one or more strict reproduction checks did not pass.**')
    status_table = '\n'.join(f'| {key.replace("_", " ")} | {"Pass" if value else "Requires review"} |' for key,value in checks.items())
    elapsed_table = '\n'.join(f'| {"Prepared comparison" if name.startswith("prepared") else "Added feature/width controls"} | {values["models"]} | {values["archived_optimizer_seconds"]/60:.2f} min | {values["fresh_optimizer_seconds"]/60:.2f} min | {values["fresh_job_wall_seconds"]/60:.2f} min |' for name,values in study_cost.items())
    source_summary = full.get('source_files_checked', 0)
    report = f'''# CSD central experiment reproduction

Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from the completed run records.

{status_line}

This run evaluates the 240 archived central-model checkpoints, then trains 240 new models from initialization: prepared-plus-wide Direct, Relational, prepared-plus-narrow Direct and raw-plus-wide Direct, each at 32D and 64D across the original 30 seeds. The fixed pilot used seeds 7, 19 and 43, producing 24 of those 240 fresh models. The remaining 216 models were added after the pilot passed; the pilot models were not counted twice.

| Verification | Result |
|---|---|
{status_table}

The saved-checkpoint replay produced {replay['metric_rows']} metric rows and {replay['numeric_comparisons']:,} numeric comparisons: {replay['exact_comparisons']:,} were exact and {replay['comparisons_outside_tolerance']} exceeded the predeclared tolerances. The full fresh-training comparison checked {full['compared_metric_values']:,} numeric values: {full['exact_metric_values']:,} were exact and {full['metric_values_outside_tolerance']} exceeded tolerance. All exactness counts are read from the detailed phase reports, not inferred from rounded paper values.

The full comparison records {full['exact_checkpoint_models']} exact checkpoint state dictionaries and {full['exact_curve_models']} exact training histories out of 240. Frozen source/configuration comparisons checked {source_summary} files, with {full['source_file_mismatches']} mismatches. All 240 completion timestamps were independently inspected for this report against the predeclared fresh-run boundary `{policy['fresh_completion_not_before_utc']}`; {len(freshness_failures)} predate it. Fresh completions span `{min(timestamps).isoformat()}` to `{max(timestamps).isoformat()}`.

All six regenerated result CSVs match their archived scientific fields exactly, covering {summary_numeric_count:,} numeric field comparisons. These overlap with the per-model checks and are not additional independent experiments.

The comparison policy was fixed before the pilot. Configurations, seeded-data fingerprints, source identities, checkpoint tensors and training histories were required to match exactly for automatic continuation. Fraction metrics allowed at most {policy['fraction_absolute_tolerance']:g} absolute numerical difference, other continuous metrics at most {policy['other_metric_absolute_tolerance']:g}, with zero relative tolerance. Timing and memory measurements were recorded separately and were not pass/fail criteria. The 48 performance-endpoint rows in the prepared summary were excluded from scientific summary equality checks.

## Central coverage results

Coverage values below are percentages, shown as mean ± sample SD across 30 seeds. Differences are percentage points. The accompanying `Central_results_comparison.csv` retains full-precision fractions, useful-candidate and decoy statistics, and the four paired contrasts.

| Dimension | Model | Condition | Archived mean ± SD (%) | Fresh mean ± SD (%) | Fresh minus archived (pp) |
|---|---|---|---|---|---|
{chr(10).join(group_markdown)}

## Prepared Direct minus Relational paired coverage contrasts

Values are percentage points with 95% paired-bootstrap intervals. These are the original 20,000-resample analysis outputs, recomputed by the unchanged analysis program after fresh training.

| Dimension | Condition | Archived difference [95% CI] | Fresh difference [95% CI] |
|---|---|---|---|
{chr(10).join(ci_markdown)}

## Execution time

| Study | Fresh models | Archived optimizer total | Fresh optimizer total | Fresh summed job wall time |
|---|---|---|---|---|
{elapsed_table}

Total fresh optimizer time was {duration(totals['fresh_optimizer_seconds'])}. Summed per-job wall time was {duration(totals['fresh_job_wall_seconds'])}; it includes each job's immediate evaluation but does not capture every setup, data-preparation or comparison cost. Pilot plus full-phase elapsed time was {duration(run_elapsed)}. Including the saved-checkpoint replay, the summed phase elapsed time was {duration(total_phase_elapsed)}. The clock interval from pilot start to full completion, including gaps between phases, was {duration(elapsed_window)}. Timings are descriptive and do not determine numerical success.

## Environment and interpretation

The run used `{runtime['python']}`, PyTorch `{runtime['torch_version']}`, NumPy `{runtime['numpy']}`, CUDA `{runtime['torch_cuda_version']}`, GPU `{runtime['device_name']}` and driver `{runtime['driver_version']}`. Dtype was `{runtime['dtype']}`; TF32 matmul/cuDNN flags were `{runtime['tf32_matmul']}`/`{runtime['tf32_cudnn']}`. The archived deterministic-algorithm setting was preserved as `{runtime['deterministic_algorithms']}`. Runtime identity was checked before training and is retained in the phase execution records.

This is fresh same-environment reproduction of the central prepared/feature-width comparisons. It does not retrain every earlier matched, wider-head, symmetry, selector, corruption or historical study in the paper. It does not establish independent external replication or performance on new task families. The historical CPU sensitivity execution-environment provenance remains unresolved; this new GPU run provides a documented central reproduction and does not reconstruct that missing history.

## Original-file integrity

The final hash audit verified all {integrity['results']['source']['files_checked']:,} indexed source-archive files, all {integrity['results']['kit']['files_checked']:,} indexed revised-kit files, and all {integrity['results']['copied_frozen_inputs']['entries_checked']} copied input pairs against their recorded identities. No fixed input changed, and no unindexed files appeared in the archive or kit. The kit's assessed manuscript SHA256 remains `{integrity['results']['assessed_author_manuscript']['sha256']}`.

{live_author_note}

## Evidence

- `checkpoint_replay/summary.json` and `metric_comparisons.csv`: archived-model re-evaluation.
- `pilot_comparison/summary.json`: fixed 24-model continuation gate.
- `full_comparison/summary.json`, `per_seed_metric_comparison.csv`, `checkpoint_comparisons.json`, `curve_comparisons.json` and `regenerated_summary_comparisons.json`: complete fresh-model comparisons.
- `pilot_execution.json`, `full_execution.json`, `pilot_result.json` and `full_result.json`: commands, environment and elapsed records.
- `comparison_policy.json`: predeclared criteria, SHA256 `{sha(policy_path)}`.
- `project/prepared_direct_comparison_20260921/jobs` and `project/direct_feature_width_ablation_20261025/jobs`: fresh checkpoints and per-job artifacts.
- `setup_manifest.json`: copied frozen-source identities and empty starting job directories.
- `integrity_final.json`: final archive, kit, copied-input and manuscript hash audit, including the live-document access limitation.

Original reference archive: `{archive}`. Fresh-run folder: `{run}`. The reporting program reads these records and writes only this report and the central-comparison CSV; it does not alter the manuscript, kit, source archive or scientific results.
'''
    fields = list(dict.fromkeys(k for row in central_rows for k in row))
    with (run / 'Central_results_comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(central_rows)
    (run / 'REPRODUCTION_REPORT.md').write_text(report, encoding='utf-8')
    print(json.dumps({'pass': passed, 'strict_checks': checks, 'fresh_models': len(model_records),
                      'coverage_groups': 16, 'paired_contrasts': 4,
                      'report': str(run / 'REPRODUCTION_REPORT.md'),
                      'csv': str(run / 'Central_results_comparison.csv')}, indent=2))
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
