"""Re-evaluate archived central-study checkpoints without optimizer updates.

Frozen scientific code is imported from --project. Every source checkpoint and
record is read from --archive. All new evidence is written under --out.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--archive', required=True, type=Path,
                        help='Full archive root containing project/')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--limit', type=int, default=None,
                        help='Optional number of models for a separate preflight output')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Optional comma-separated original seed subset')
    return parser.parse_args()


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = Path(path).with_suffix('.tmp.csv')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def tolerance(metric):
    if metric in ('parameter_count', 'output_raw_rank', 'output_direction_rank', 'decoder_augmented_rank'):
        return 0.0
    if 'angular_error_degrees' in metric:
        return 0.02
    if any(word in metric for word in ('recall', 'rate', 'fraction')):
        return 1e-6
    return 2e-5


def main():
    args = parse_args()
    project = args.project.resolve(strict=True)
    archive = args.archive.resolve(strict=True)
    out = args.out.resolve()
    if out == archive or out.is_relative_to(archive):
        raise ValueError('Output must not be inside the original archive')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new empty replay output directory; records are never silently overwritten')
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be positive')
    out.mkdir(parents=True, exist_ok=True)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True

    sys.path.insert(0, str(project/'prepared_direct_comparison_20260921'))
    import experiment as prior
    import torch
    import numpy as np
    spec = importlib.util.spec_from_file_location('csd_width_checkpoint_replay', project/'direct_feature_width_ablation_20261025/run.py')
    width = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(width)
    assert prior.OUT.resolve() == project/'prepared_direct_comparison_20260921'
    assert width.PRIOR.resolve() == prior.OUT.resolve()
    prior.check_lock()
    width.check_lock()

    prepared = archive/'project/prepared_direct_comparison_20260921'
    controls = archive/'project/direct_feature_width_ablation_20261025'
    baseline = load(prepared/'runtime.json')
    baseline_width = load(controls/'runtime.json')
    # Preserve recorded CPU threading for CPU diagnostic SVD and reductions.
    torch.set_num_threads(baseline['threads'])
    torch.set_num_interop_threads(baseline['interop_threads'])
    device = prior.configure_runtime('cuda')
    runtime = prior.runtime(device)
    runtime['assessment_utc'] = stamp()
    runtime['operation'] = 'saved_checkpoint_forward_evaluation_no_training'
    runtime['checkpoint_driver_sha256'] = sha(Path(__file__))
    runtime['working_project'] = str(project)
    runtime['original_archive'] = str(archive)
    scientific_keys = ('torch_version','torch_cuda_version','device','device_name','dtype','tf32_matmul','tf32_cudnn','numpy','threads','interop_threads','cudnn_version','deterministic_algorithms','matmul_precision','driver_version')
    differences = {key: {'archive': baseline.get(key), 'current': runtime.get(key)}
                   for key in scientific_keys if baseline.get(key) != runtime.get(key)}
    runtime['archived_runtime_differences'] = differences
    runtime['archived_executable'] = baseline.get('executable')
    runtime['archived_python_build'] = baseline.get('python')
    write(out/'runtime.json', runtime)
    # The caller must explicitly inspect an environment mismatch before inference.
    assert not differences, ('Scientific runtime differs from archived primary environment', differences)
    assert all(baseline.get(key) == baseline_width.get(key) for key in scientific_keys), 'Original two-study runtime identities differ'

    seeds = list(prior.matched_config(32).seeds)
    assert seeds == list(prior.matched_config(64).seeds) and len(seeds) == 30
    if args.seeds:
        requested = [int(s.strip()) for s in args.seeds.split(',')]
        assert len(requested) == len(set(requested)) and set(requested).issubset(seeds)
        seeds = requested
    kinds = ('DirectPrepared','Relational','DirectPreparedNarrow','DirectRawWide')
    planned = [(d, seed, kind) for d in (32,64) for seed in seeds for kind in kinds]
    if args.limit is not None:
        planned = planned[:args.limit]
    plan = {'created_utc':stamp(), 'models':len(planned), 'jobs':[{'dimension':d,'seed':s,'model':k} for d,s,k in planned],
            'splits':['iid','exact_rotated'], 'training_fit_recomputed':True,
            'checkpoint_inference':True, 'optimizer_updates':False, 'fresh_training':False,
            'criteria':{'ranks_and_parameter_count_absolute':0, 'recall_rate_fraction_absolute':1e-6,
                        'angular_error_degrees_absolute':0.02, 'other_numeric_absolute':2e-5,
                        'basis':'Original prepared historic() validator, with stricter exact ranks/counts and 1e-6 rate/fraction flags.',
                        'meaning':'Predeclared comparison flags; all exactness and raw deltas are retained. No tolerance is widened after viewing results.'},
            'source_sha256':{'prepared_lock':sha(prepared/'lock.json'),'width_lock':sha(controls/'lock.json'),
                             'working_prepared_lock':sha(prior.OUT/'lock.json'),'working_width_lock':sha(width.HERE/'lock.json'),
                             'working_prepared_runner':sha(prior.OUT/'experiment.py'),'working_width_runner':sha(width.HERE/'run.py')}}
    assert plan['source_sha256']['prepared_lock'] == plan['source_sha256']['working_prepared_lock']
    assert plan['source_sha256']['width_lock'] == plan['source_sha256']['working_width_lock']
    write(out/'plan.json',plan)

    all_metrics = []
    comparisons = []
    jobs = []
    data_checks = []
    started = time.perf_counter()
    expected_by_pair = {}
    for d, s, k in planned:
        expected_by_pair.setdefault((d,s),[]).append(k)

    def progress(state, **extra):
        write(out/'progress.json',{'utc':stamp(),'state':state,'completed_models':len(jobs),'planned_models':len(planned),
                                  'elapsed_seconds':time.perf_counter()-started,**extra})

    progress('running')
    try:
        for (d, seed), selected_kinds in expected_by_pair.items():
            pair_start = time.perf_counter()
            progress('generating_tasks',dimension=d,seed=seed)
            c,b,tr,h,eq,prepared_components,stats,splits,feature_hashes,prep_seconds = prior.data(d,seed,device,save=False)
            provenance_path = prepared/'data'/f'd{d}_s{seed}'/'provenance.json'
            provenance = load(provenance_path)
            records_path = provenance_path.with_name('records.npz')
            assert feature_hashes == provenance['feature_hashes'], ('Regenerated feature/data/normalization hashes differ',d,seed)
            assert sha(records_path) == provenance['records_sha256'], ('Archived task arrays changed',d,seed)
            # Compare saved starting arrays directly with regenerated tensors.
            with np.load(records_path) as records:
                assert np.array_equal(records['train_starts'].reshape(-1,d), h.detach().cpu().numpy()), ('Training starts differ',d,seed)
                assert np.array_equal(records['test_starts'].reshape(-1,d), splits['iid'][1].detach().cpu().numpy()), ('Test starts differ',d,seed)
            raw = prior.build_component_descriptors(tr,h) if any(k in ('DirectPreparedNarrow','DirectRawWide') for k in selected_kinds) else None
            raw_stats = prior.descriptor_normalization(raw) if raw is not None else None
            assert raw is None or torch.equal(raw.mask,prepared_components.mask)
            data_checks.append({'dimension':d,'seed':seed,'feature_hashes_match':True,'archived_records_sha256':sha(records_path),
                                'saved_start_arrays_match':True,'feature_hashes':feature_hashes,'data_preparation_seconds':prep_seconds})
            for kind in selected_kinds:
                progress('evaluating_checkpoint',dimension=d,seed=seed,model=kind)
                source_study = prepared if kind in prior.MODELS else controls
                folder = source_study/'jobs'/f'd{d}'/f'seed_{seed}'/kind
                completion = load(folder/'completion.json')
                key=f'd{d}_s{seed}_{kind}'
                assert completion['key']==key and completion['steps']==1500 and completion['status']=='complete'
                for filename,digest in completion['files'].items():
                    assert sha(folder/filename)==digest, ('Original job integrity mismatch',key,filename)
                metadata=load(folder/'metadata.json')
                assert metadata['feature_hashes']==feature_hashes
                payload=torch.load(folder/'checkpoint.pt',map_location=device,weights_only=True)
                state=payload['state_dict']
                if kind=='Relational':
                    normalization=stats['Relational']; comp=eq; alias=kind
                    model=prior.factory(kind,c,seed,device,normalization)
                    buffer_names=('invariant_mean','invariant_scale')
                elif kind=='DirectPrepared':
                    normalization=stats['DirectPrepared']; comp=prepared_components; alias=kind
                    model=prior.factory(kind,c,seed,device,normalization)
                    buffer_names=('descriptor_mean','descriptor_scale')
                else:
                    normalization=stats['DirectPrepared'] if kind=='DirectPreparedNarrow' else raw_stats
                    comp=prepared_components if kind=='DirectPreparedNarrow' else raw
                    alias='DirectPrepared' if kind=='DirectPreparedNarrow' else 'HistoricalDirect'
                    model=width.model_for(kind,c,seed,device,normalization)
                    buffer_names=('descriptor_mean','descriptor_scale')
                for name,expected in zip(buffer_names,normalization):
                    assert torch.equal(state[name],expected), ('Checkpoint normalization differs from regenerated training normalization',key,name)
                model.load_state_dict(state,strict=True)
                model.eval()
                parameter_count=sum(p.numel() for p in model.parameters())
                assert parameter_count==metadata['parameter_count']
                fit=prior.training_fit(model,tr,h,comp,c)
                archived_rows={r['split']:r for r in load(folder/'metrics.json')}
                assert set(archived_rows)==set(splits)
                model_comparisons=[]
                for split,(tasks,starts) in splits.items():
                    metric,diagnostic=prior.evaluate(alias,model,tasks,starts,b,c)
                    metric['parameter_count']=parameter_count
                    metric.update({'train_'+name:value for name,value in fit.items()})
                    actual={'dimension':d,'seed':seed,'model':kind,'split':split,**metric}
                    all_metrics.append(actual)
                    reference=archived_rows[split]
                    assert all(name in reference for name in metric), ('Recomputed metric absent from original',key,split,set(metric)-set(reference))
                    for name,actual_value in metric.items():
                        old=float(reference[name]); new=float(actual_value)
                        assert math.isfinite(old) and math.isfinite(new)
                        absolute=abs(new-old); bound=tolerance(name)
                        entry={'dimension':d,'seed':seed,'model':kind,'split':split,'metric':name,
                               'archived_value':old,'replayed_value':new,'signed_delta':new-old,'absolute_delta':absolute,
                               'relative_delta':absolute/max(abs(old),1e-300),'exact':new==old,'absolute_tolerance':bound,
                               'within_predeclared_tolerance':absolute<=bound}
                        comparisons.append(entry);model_comparisons.append(entry)
                prior.synchronize(device)
                jobs.append({'dimension':d,'seed':seed,'model':kind,'checkpoint_sha256':sha(folder/'checkpoint.pt'),
                             'normalization_matches_regeneration':True,'numeric_comparisons':len(model_comparisons),
                             'all_exact':all(r['exact'] for r in model_comparisons),
                             'all_within_tolerance':all(r['within_predeclared_tolerance'] for r in model_comparisons)})
                write_csv(out/'replayed_metrics.csv',all_metrics)
                write_csv(out/'metric_comparisons.csv',comparisons)
                write(out/'model_checks.json',jobs)
                write(out/'data_checks.json',data_checks)
                print(f'{stamp()} REPLAY {len(jobs)}/{len(planned)} {key} exact={jobs[-1]["all_exact"]} tolerance={jobs[-1]["all_within_tolerance"]}',flush=True)
                del model,payload,state,comp,normalization
                gc.collect();torch.cuda.empty_cache()
            del c,b,tr,h,eq,prepared_components,stats,splits,raw,raw_stats
            gc.collect();torch.cuda.empty_cache()
            progress('pair_complete',dimension=d,seed=seed,pair_seconds=time.perf_counter()-pair_start)
        outside=[r for r in comparisons if not r['within_predeclared_tolerance']]
        result={'completed_utc':stamp(),'models_evaluated':len(jobs),'metric_rows':len(all_metrics),
                'numeric_comparisons':len(comparisons),'exact_comparisons':sum(r['exact'] for r in comparisons),
                'comparisons_outside_tolerance':len(outside),'all_within_predeclared_tolerance':not outside,
                'all_exact':all(r['exact'] for r in comparisons),'max_absolute_delta':max((r['absolute_delta'] for r in comparisons),default=0),
                'models_all_exact':sum(r['all_exact'] for r in jobs),'data_pairs':len(data_checks),
                'elapsed_seconds':time.perf_counter()-started,'inference_performed':True,'training_performed':False,
                'completion_status':'pass' if not outside else 'requires_review'}
        write(out/'outside_tolerance.json',outside)
        write(out/'summary.json',result)
        progress('complete' if not outside else 'requires_review')
        print(json.dumps(result,indent=2),flush=True)
        return 0 if not outside else 2
    except BaseException as error:
        write(out/'failure.json',{'utc':stamp(),'error':str(error),'traceback':traceback.format_exc()})
        progress('failed',error=str(error))
        raise


if __name__=='__main__':
    raise SystemExit(main())
