"""CPU-only comparison of isolated fresh central-study jobs against the archive.

Does not import or execute scientific training/evaluation code. Reads checkpoints
with weights_only=True on CPU, and never changes either input project.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

STUDIES = {
    'prepared_direct_comparison_20260921': ('DirectPrepared', 'Relational'),
    'direct_feature_width_ablation_20261025': ('DirectPreparedNarrow', 'DirectRawWide'),
}
NON_SCIENTIFIC_FIELDS = {
    'optimizer_seconds', 'job_wall_seconds', 'training_peak_allocated_bytes',
    'training_peak_reserved_bytes', 'elapsed_seconds', 'training_seconds',
}
IDENTITY_FIELDS = {'dimension', 'seed', 'model', 'split', 'status'}
INTEGER_FIELDS = {'parameter_count', 'output_raw_rank', 'output_direction_rank', 'decoder_augmented_rank'}
FRACTION_FIELDS = {
    'proposal_recall', 'mode_set_recall', 'target_proposal_recall',
    'candidate_decoy_rate', 'candidate_anchor_rate', 'useful_candidate_fraction',
    'attention_useful_mass', 'attention_target_specialization', 'useful_region_count',
}
EXPECTED_FILES = {'checkpoint.pt', 'configuration.json', 'curves.json', 'metadata.json',
                  'metrics.json', 'rank_diagnostics.json', 'training_fit.json', 'verification.json'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def numeric(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def finite(v):
    if numeric(v):
        return math.isfinite(v)
    if isinstance(v, dict):
        return all(finite(x) for x in v.values())
    if isinstance(v, list):
        return all(finite(x) for x in v)
    return True


def flatten(v, prefix=''):
    if isinstance(v, dict):
        for k in sorted(v):
            yield from flatten(v[k], prefix + '/' + str(k))
    elif isinstance(v, list):
        for i, item in enumerate(v):
            yield from flatten(item, prefix + '/' + str(i))
    else:
        yield prefix, v


def exact_json_comparison(a, b):
    fa, fb = dict(flatten(a)), dict(flatten(b))
    differences = []
    max_abs = 0.0
    for key in sorted(fa.keys() | fb.keys()):
        av, bv = fa.get(key), fb.get(key)
        if key not in fa or key not in fb or av != bv:
            delta = abs(float(av) - float(bv)) if numeric(av) and numeric(bv) else None
            if delta is not None and math.isfinite(delta):
                max_abs = max(max_abs, delta)
            differences.append({'field': key, 'archived': av, 'fresh': bv, 'absolute_difference': delta})
    return {'exact': not differences, 'different_fields': len(differences),
            'max_absolute_numeric_difference': max_abs, 'differences': differences}


def checkpoint_compare(a, b, torch):
    old = torch.load(a, map_location='cpu', weights_only=True)
    new = torch.load(b, map_location='cpu', weights_only=True)
    sa, sb = old['state_dict'], new['state_dict']
    diffs = []
    count = 0
    elements = 0
    for key in sorted(sa.keys() | sb.keys()):
        if key not in sa or key not in sb:
            diffs.append({'tensor': key, 'reason': 'missing tensor'})
            continue
        av, bv = sa[key], sb[key]
        count += 1
        elements += av.numel()
        if av.shape != bv.shape or av.dtype != bv.dtype:
            diffs.append({'tensor': key, 'reason': 'shape or dtype differs',
                          'archived_shape': list(av.shape), 'fresh_shape': list(bv.shape),
                          'archived_dtype': str(av.dtype), 'fresh_dtype': str(bv.dtype)})
        elif not torch.equal(av, bv):
            delta = (av.to(torch.float64) - bv.to(torch.float64)).abs()
            diffs.append({'tensor': key, 'reason': 'values differ',
                          'different_elements': int(torch.count_nonzero(delta)),
                          'max_absolute_difference': float(delta.max()),
                          'rms_difference': float(torch.sqrt(torch.mean(delta.square())))})
    history = exact_json_comparison(old.get('history'), new.get('history'))
    return {'tensors_exact': not diffs, 'tensors_checked': count, 'elements_checked': elements,
            'tensor_differences': diffs, 'history': history}


def validate_completion(job, expected_key, lock, steps, not_before=None):
    errors = []
    p = job / 'completion.json'
    if not p.is_file():
        return ['completion.json missing']
    obj = read(p)
    if obj.get('key') != expected_key or obj.get('status') != 'complete' or obj.get('steps') != steps:
        errors.append('Completion identity/status/steps mismatch')
    if set(obj.get('files', {})) != EXPECTED_FILES:
        errors.append('Completion-listed files differ from the expected eight files')
    for name, digest in obj.get('files', {}).items():
        path = (job / name).resolve()
        if not path.is_relative_to(job.resolve()):
            errors.append('Completion path outside job folder')
        elif not path.is_file() or sha(path) != digest:
            errors.append('Completion hash mismatch: ' + name)
    allowed = {sha(lock)} | set(read(lock).get('compatible_prior_locks', {}))
    if obj.get('lock_sha256') not in allowed:
        errors.append('Unknown source-lock identity')
    if not_before:
        try:
            when = datetime.fromisoformat(obj['completed_utc'].replace('Z', '+00:00'))
            limit = datetime.fromisoformat(not_before.replace('Z', '+00:00'))
            if when < limit:
                errors.append('Completion predates the fresh reproduction session')
        except (KeyError, ValueError):
            errors.append('Cannot validate fresh completion timestamp')
    return errors


def source_check(old_project, new_project):
    files = []
    prepared = 'prepared_direct_comparison_20260921'
    width = 'direct_feature_width_ablation_20261025'
    p = old_project / prepared
    files += [f'{prepared}/{n}' for n in ['PLAN.md', 'experiment.py', 'analyze.py', 'configurations.json', 'lock.json']]
    files += [f'{prepared}/{f.relative_to(p).as_posix()}' for f in sorted((p / 'source_snapshot').rglob('*.py'))]
    files += [f'{width}/{n}' for n in ['PLAN.md', 'run.py', 'lock.json', 'configuration.json']]
    # Preserve every archived source/lock referred to by compatible-prior-lock records.
    for study in STUDIES:
        lock = read(old_project / study / 'lock.json')
        for record in lock.get('compatible_prior_locks', {}).values():
            for key in ('lock_path', 'runner_path', 'source_path'):
                if key in record:
                    files.append(f'{study}/{record[key]}')
    out = []
    for rel in sorted(set(files)):
        old, new = old_project / rel, new_project / rel
        out.append({'path': rel, 'archived_sha256': sha(old) if old.is_file() else None,
                    'fresh_sha256': sha(new) if new.is_file() else None,
                    'exact': old.is_file() and new.is_file() and sha(old) == sha(new)})
    return out


def compare_summaries(old_project, new_project):
    results = []
    files = {
        'prepared_direct_comparison_20260921': ['per_seed.csv', 'comparison_summary.csv'],
        'direct_feature_width_ablation_20261025': ['per_seed.csv', 'comparison_summary.csv', 'contrasts.csv', 'new_pair_contrasts.csv'],
    }
    for study, names in files.items():
        for name in names:
            a, b = old_project / study / name, new_project / study / name
            if not b.is_file():
                results.append({'file': f'{study}/{name}', 'available': False})
                continue
            with a.open(encoding='utf-8-sig', newline='') as f:
                old = list(csv.DictReader(f))
            with b.open(encoding='utf-8-sig', newline='') as f:
                new = list(csv.DictReader(f))
            original_counts = {'archived': len(old), 'fresh': len(new)}
            skipped_old = [r for r in old if r.get('endpoint', r.get('metric')) in NON_SCIENTIFIC_FIELDS]
            skipped_new = [r for r in new if r.get('endpoint', r.get('metric')) in NON_SCIENTIFIC_FIELDS]
            old = [r for r in old if r.get('endpoint', r.get('metric')) not in NON_SCIENTIFIC_FIELDS]
            new = [r for r in new if r.get('endpoint', r.get('metric')) not in NON_SCIENTIFIC_FIELDS]
            errors = []
            max_delta = 0.0
            fields = 0
            text_fields = 0
            for i, (ra, rb) in enumerate(zip(old, new)):
                for key in ra.keys() | rb.keys():
                    if key in NON_SCIENTIFIC_FIELDS:
                        continue
                    av, bv = ra.get(key), rb.get(key)
                    try:
                        delta = abs(float(av) - float(bv))
                        fields += 1
                        if not math.isfinite(delta) or delta != 0:
                            errors.append({'row': i + 1, 'field': key, 'old': av, 'fresh': bv, 'abs_difference': delta})
                            max_delta = max(max_delta, delta)
                    except (TypeError, ValueError):
                        text_fields += 1
                        if av != bv:
                            errors.append({'row': i + 1, 'field': key, 'old': av, 'fresh': bv})
            results.append({'file': f'{study}/{name}', 'available': True, 'archived_rows': len(old),
                            'fresh_rows': len(new), 'exact_scientific_values': len(old) == len(new) and not errors,
                            'total_rows_before_performance_filter': original_counts,
                            'excluded_performance_rows': {'archived': len(skipped_old), 'fresh': len(skipped_new)},
                            'excluded_performance_endpoints': sorted({r.get('endpoint', r.get('metric')) for r in skipped_old + skipped_new}),
                            'numeric_fields_compared': fields, 'text_fields_compared': text_fields,
                            'different_fields': len(errors), 'max_absolute_numeric_difference': max_delta,
                            'differences': errors})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path, help='Full original research archive, with project/')
    parser.add_argument('--project', required=True, type=Path, help='Fresh working project containing the two studies')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--policy', required=True, type=Path)
    parser.add_argument('--stage', choices=['pilot', 'full'], required=True)
    parser.add_argument('--skip-tensors', action='store_true', help='Diagnostic only: makes strict automatic gate false')
    args = parser.parse_args()
    archive, project, out = args.archive.resolve(), args.project.resolve(), args.out.resolve()
    old_project = archive / 'project'
    if project == old_project or project.is_relative_to(archive) or out.is_relative_to(archive):
        raise ValueError('Fresh project/output must be outside the original archive')
    out.mkdir(parents=True, exist_ok=True)
    policy = read(args.policy)
    assert policy['relative_tolerance'] == 0, 'Driver implements declared absolute tolerances only'
    all_seeds = read(old_project / 'prepared_direct_comparison_20260921/configurations.json')['32']['matched']['seeds']
    seeds = policy['pilot_seeds'] if args.stage == 'pilot' else all_seeds
    assert len(seeds) == len(set(seeds)) and set(seeds) <= set(all_seeds)
    torch = None
    if not args.skip_tensors:
        import torch
        torch.set_num_threads(1)

    source = source_check(old_project, project)
    global_issues = ['Frozen source/config differs: ' + x['path'] for x in source if not x['exact']]
    if args.skip_tensors:
        global_issues.append('Checkpoint tensor comparison skipped')
    if not policy.get('fresh_completion_not_before_utc'):
        global_issues.append('Policy has no fresh completion timestamp boundary; freshness requires review')
    jobs, metrics, timings, tensor_details, curve_details = [], [], [], [], []
    aggregate = defaultdict(lambda: [[], []])
    runtime_records = []
    provenance = []
    for study, kinds in STUDIES.items():
        old_runtime = read(old_project / study / 'runtime.json') if (old_project / study / 'runtime.json').exists() else {}
        new_runtime = read(project / study / 'runtime.json') if (project / study / 'runtime.json').exists() else {}
        for d in policy['dimensions']:
            for seed in seeds:
                for model in kinds:
                    key = f'd{d}_s{seed}_{model}'
                    rel = Path(study) / 'jobs' / f'd{d}' / f'seed_{seed}' / model
                    old, new = old_project / rel, project / rel
                    issues = []
                    if not new.is_dir():
                        jobs.append({'key': key, 'study': study, 'dimension': d, 'seed': seed, 'model': model,
                                     'present': False, 'passed': False, 'issues': ['Fresh job is missing']})
                        continue
                    issues.extend(validate_completion(new, key, project / study / 'lock.json', policy['expected_optimizer_steps'], policy.get('fresh_completion_not_before_utc')))
                    old_errors = validate_completion(old, key, old_project / study / 'lock.json', policy['expected_optimizer_steps'])
                    issues.extend('Archived reference: ' + s for s in old_errors)
                    required = ['configuration.json', 'metadata.json', 'curves.json', 'metrics.json', 'verification.json', 'checkpoint.pt']
                    if any(not (new / name).is_file() for name in required):
                        jobs.append({'key': key, 'study': study, 'dimension': d, 'seed': seed, 'model': model,
                                     'present': True, 'passed': False, 'issues': issues + ['Required job artifact missing']})
                        continue
                    oc, nc = read(old / 'configuration.json'), read(new / 'configuration.json')
                    if oc != nc:
                        issues.append('Scientific job configuration differs')
                    om, nm = read(old / 'metadata.json'), read(new / 'metadata.json')
                    if om.get('feature_hashes') != nm.get('feature_hashes'):
                        issues.append('Seeded data/feature/normalization fingerprints differ')
                    if not nm.get('feature_hashes'):
                        issues.append('Seeded feature hashes are missing')
                    for field in ('dimension', 'seed', 'model', 'parameter_count', 'steps', 'batch_size', 'examples_seen'):
                        if field in om and nm.get(field) != om[field]:
                            issues.append('Model/training identity differs: ' + field)
                    for field in NON_SCIENTIFIC_FIELDS:
                        if field in om or field in nm:
                            timings.append({'key': key, 'field': field, 'archived': om.get(field), 'fresh': nm.get(field)})
                    oruntime = dict(old_runtime, **om.get('runtime', {}))
                    nruntime = dict(new_runtime, **nm.get('runtime', {}))
                    for field in policy['runtime_identity_keys']:
                        if field not in oruntime:
                            continue
                        same = oruntime.get(field) == nruntime.get(field)
                        allowance = policy.get('allowed_runtime_differences', {}).get(field, '')
                        runtime_records.append({'key': key, 'field': field, 'archived': oruntime.get(field),
                                                'fresh': nruntime.get(field), 'exact': same, 'predeclared_allowance': allowance})
                        if not same and not allowance:
                            issues.append('Runtime identity differs or is missing: ' + field)
                    verify = read(new / 'verification.json')
                    if not finite(nm) or not finite(read(new / 'metrics.json')) or not finite(read(new / 'curves.json')):
                        issues.append('Nonfinite value in recorded model/metrics/history')
                    if verify.get('checkpoint_round_trip_exact_metrics') is False or verify.get('finite_parameters_and_logged_loss_gradients') is False:
                        issues.append('Original runner verification failed')
                    oa, na = read(old / 'metrics.json'), read(new / 'metrics.json')
                    old_lookup = {r['split']: r for r in oa}
                    new_lookup = {r['split']: r for r in na}
                    if len(oa) != 2 or len(na) != 2 or set(old_lookup) != {'iid', 'exact_rotated'} or set(new_lookup) != {'iid', 'exact_rotated'}:
                        issues.append('Expected two unique IID/rotation metric rows')
                    scientific_exact = True
                    outside = 0
                    for split in sorted(old_lookup.keys() & new_lookup.keys()):
                        ra, rb = old_lookup[split], new_lookup[split]
                        for field in (ra.keys() | rb.keys()) - NON_SCIENTIFIC_FIELDS:
                            av, bv = ra.get(field), rb.get(field)
                            if field in IDENTITY_FIELDS:
                                if av != bv:
                                    issues.append('Result identity differs: ' + field)
                                continue
                            if not numeric(av) or not numeric(bv):
                                if av != bv:
                                    issues.append('Result field differs/is missing: ' + field)
                                continue
                            tolerance = (0.0 if field in INTEGER_FIELDS else
                                         policy['fraction_absolute_tolerance'] if field in FRACTION_FIELDS else
                                         policy['other_metric_absolute_tolerance'])
                            delta = abs(float(av) - float(bv))
                            exact = av == bv
                            passed = math.isfinite(delta) and delta <= tolerance
                            scientific_exact &= exact
                            outside += not passed
                            metrics.append({'key': key, 'study': study, 'dimension': d, 'seed': seed, 'model': model,
                                            'split': split, 'metric': field, 'archived': av, 'fresh': bv,
                                            'absolute_difference': delta, 'tolerance': tolerance, 'exact': exact,
                                            'within_declared_tolerance': passed})
                            aggregate[(study, d, model, split, field)][0].append(float(av))
                            aggregate[(study, d, model, split, field)][1].append(float(bv))
                    if outside:
                        issues.append(f'{outside} scientific metric values exceed declared tolerance')
                    curve = exact_json_comparison(read(old / 'curves.json'), read(new / 'curves.json'))
                    curve_details.append({'key': key, **curve})
                    if not curve['exact'] and policy['training_histories_exact_for_automatic_continuation']:
                        issues.append('Training curve differs from archived history')
                    tensors_exact = None
                    if torch is not None:
                        cp = checkpoint_compare(old / 'checkpoint.pt', new / 'checkpoint.pt', torch)
                        tensor_details.append({'key': key, **cp})
                        tensors_exact = cp['tensors_exact']
                        if not tensors_exact and policy['checkpoint_tensors_exact_for_automatic_continuation']:
                            issues.append('Checkpoint state tensors differ')
                        if not cp['history']['exact'] and policy['training_histories_exact_for_automatic_continuation']:
                            issues.append('Checkpoint training history differs')
                    # Compare generated provenance once per dimension/seed; both studies share it.
                    if model == 'DirectPrepared':
                        p = Path('prepared_direct_comparison_20260921/data') / f'd{d}_s{seed}' / 'provenance.json'
                        if (project / p).is_file():
                            comparison = exact_json_comparison(read(old_project / p), read(project / p))
                            provenance.append({'dimension': d, 'seed': seed, **comparison})
                            if not comparison['exact']:
                                issues.append('Generated task/feature provenance differs')
                        else:
                            issues.append('Fresh generated data provenance missing')
                    jobs.append({'key': key, 'study': study, 'dimension': d, 'seed': seed, 'model': model,
                                 'present': True, 'scientific_metrics_exact': scientific_exact,
                                 'scientific_metrics_outside_tolerance': outside, 'curve_exact': curve['exact'],
                                 'checkpoint_tensors_exact': tensors_exact, 'passed': not issues, 'issues': issues})

    agg_rows = []
    for (study, d, model, split, field), (a, b) in sorted(aggregate.items()):
        agg_rows.append({'study': study, 'dimension': d, 'model': model, 'split': split, 'metric': field,
                         'n_seeds': len(a), 'archived_mean': statistics.mean(a), 'fresh_mean': statistics.mean(b),
                         'mean_difference': statistics.mean(b) - statistics.mean(a),
                         'archived_sample_sd': statistics.stdev(a) if len(a) > 1 else 0,
                         'fresh_sample_sd': statistics.stdev(b) if len(b) > 1 else 0})
    summaries = compare_summaries(old_project, project) if args.stage == 'full' else []
    if args.stage == 'full' and policy.get('require_full_summary_files'):
        for item in summaries:
            if not item.get('available') or not item.get('exact_scientific_values'):
                global_issues.append('Full regenerated summary requires review: ' + item['file'])
    expected = len(seeds) * len(policy['dimensions']) * 4
    passed = len(jobs) == expected and all(j['passed'] for j in jobs) and not global_issues
    summary = {
        'stage': args.stage, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'archive': str(archive), 'fresh_project': str(project), 'policy_path': str(args.policy.resolve()),
        'policy_sha256': sha(args.policy), 'policy': policy,
        'expected_models': expected, 'completed_models': sum(j['present'] for j in jobs),
        'passing_models': sum(j['passed'] for j in jobs), 'seeds': seeds,
        'source_files_checked': len(source), 'source_file_mismatches': sum(not r['exact'] for r in source),
        'compared_metric_values': len(metrics), 'exact_metric_values': sum(r['exact'] for r in metrics),
        'metric_values_outside_tolerance': sum(not r['within_declared_tolerance'] for r in metrics),
        'exact_checkpoint_models': sum(j.get('checkpoint_tensors_exact') is True for j in jobs),
        'exact_curve_models': sum(j.get('curve_exact') is True for j in jobs),
        'global_issues': global_issues, 'jobs_requiring_review': [j for j in jobs if not j['passed']],
        'automatic_continuation_pass': passed, 'pass': passed,
        'interpretation': policy['interpretation'],
        'comparison_environment': {'python': sys.version, 'torch': torch.__version__ if torch else None, 'device': 'cpu'},
    }
    write(out / 'summary.json', summary)
    write(out / 'job_checks.json', jobs)
    write(out / 'source_checks.json', source)
    write(out / 'checkpoint_comparisons.json', tensor_details)
    write(out / 'curve_comparisons.json', curve_details)
    write(out / 'provenance_comparisons.json', provenance)
    write(out / 'runtime_comparisons.json', runtime_records)
    write(out / 'regenerated_summary_comparisons.json', summaries)
    csv_write(out / 'per_seed_metric_comparison.csv', metrics)
    csv_write(out / 'aggregate_comparison.csv', agg_rows)
    csv_write(out / 'timing_and_memory.csv', timings)
    print(json.dumps({k: summary[k] for k in ['stage', 'expected_models', 'completed_models', 'passing_models',
          'compared_metric_values', 'exact_metric_values', 'metric_values_outside_tolerance',
          'exact_checkpoint_models', 'exact_curve_models', 'global_issues', 'pass']}, indent=2))
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
