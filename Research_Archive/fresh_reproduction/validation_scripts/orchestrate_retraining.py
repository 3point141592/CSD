"""Run unmodified frozen training functions in an isolated reproduction project."""
from __future__ import annotations
import argparse, gc, hashlib, importlib.util, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def stamp(): return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,p)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--policy',type=Path,required=True)
    ap.add_argument('--comparator',type=Path,required=True)
    ap.add_argument('--phase',choices=['pilot','full','analyze'],required=True)
    args=ap.parse_args()
    project=args.project.resolve(strict=True);archive=args.archive.resolve(strict=True)
    run=project.parent;policy=load(args.policy)
    assert project!=archive and not project.is_relative_to(archive)
    replay=load(run/'checkpoint_replay/summary.json')
    assert replay['models_evaluated']==240 and replay['all_within_predeclared_tolerance'], 'Checkpoint replay must pass before fresh training'
    if args.phase=='full':
        pilot=load(run/'pilot_comparison/summary.json')
        assert pilot['pass'] and pilot['completed_models']==24 and pilot['policy_sha256']==sha(args.policy), 'Fixed24-model pilot must pass before expansion'
    assert policy['fresh_completion_not_before_utc']
    sys.dont_write_bytecode=True;os.environ['PYTHONDONTWRITEBYTECODE']='1'
    sys.path.insert(0,str(project/'prepared_direct_comparison_20260921'))
    import experiment as prepared
    import torch, psutil
    spec=importlib.util.spec_from_file_location('csd_fresh_width',project/'direct_feature_width_ablation_20261025/run.py')
    width=importlib.util.module_from_spec(spec);spec.loader.exec_module(width)
    prepared.check_lock();width.check_lock()
    torch.set_num_threads(24);torch.set_num_interop_threads(24)
    device=prepared.configure_runtime('cuda')
    actual=prepared.runtime(device)
    baseline=load(archive/'project/prepared_direct_comparison_20260921/runtime.json')
    runtime_diff={k:{'archived':baseline.get(k),'fresh':actual.get(k)} for k in policy['runtime_identity_keys'] if baseline.get(k)!=actual.get(k)}
    assert not runtime_diff, ('Scientific runtime identity differs',runtime_diff)
    assert shutil.disk_usage(run).free>15*1024**3,'At least15GiB free disk required before starting'
    assert not width.external_gpu_job(), 'Another recognized GPU training job is active'
    global_lock=run/'reproduction_worker.json'
    if global_lock.exists():
        old=load(global_lock)
        if psutil.pid_exists(old['pid']) and abs(psutil.Process(old['pid']).create_time()-old['process_create_time'])<1:
            raise RuntimeError('A reproduction worker is already active')
        os.replace(global_lock,run/f'reproduction_worker_stale_{int(time.time())}.json')
    with global_lock.open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'process_create_time':psutil.Process().create_time(),'phase':args.phase,'started_utc':stamp()},f,indent=2)
    held_locks=[global_lock]
    try:
        prepared_lock=prepared.acquire_worker_lock();held_locks.append(prepared_lock)
        width_lock=width.HERE/'worker.lock.json'
        assert not width_lock.exists(),'Width worker lock must be absent'
        write(width_lock,{'pid':os.getpid(),'utc':stamp(),'orchestrator_phase':args.phase});held_locks.append(width_lock)
        prepared.verify_runtime_identity(actual)
        if (width.HERE/'runtime.json').exists():
            old=load(width.HERE/'runtime.json');assert all(old.get(k)==actual.get(k) for k in policy['runtime_identity_keys'])
        else:write(width.HERE/'runtime.json',actual)
        write(run/f'{args.phase}_execution.json',{'started_utc':stamp(),'command':sys.argv,'python_executable':sys.executable,'runtime':actual,'orchestrator_sha256':sha(Path(__file__)),'comparison_policy_sha256':sha(args.policy),'phase':args.phase,'fresh_training':args.phase!='analyze'})
    except BaseException:
        for path in held_locks:
            if path.exists():os.replace(path,path.with_name(path.name+f'.ended_setup_{int(time.time())}'))
        raise
    started=time.perf_counter()
    def progress(state,**extra):
        counts={}
        for name,module in [('prepared',prepared),('feature_width',width)]:
            folder=module.OUT if name=='prepared' else module.HERE
            m=load(folder/'manifest.json')
            counts[name]=sum(v.get('status')=='complete' for v in m['jobs'].values())
        write(run/'run_status.json',{'utc':stamp(),'phase':args.phase,'state':state,'completed_models':sum(counts.values()),'by_study':counts,'elapsed_seconds':time.perf_counter()-started,**extra})
    try:
        if args.phase=='pilot':
            for folder in [prepared.OUT,width.HERE]:
                assert not list((folder/'jobs').glob('**/completion.json')), 'Pilot must start with no completed fresh jobs'
            prepared.verify(device)
        if args.phase!='analyze':
            seeds=policy['pilot_seeds'] if args.phase=='pilot' else list(prepared.matched_config(32).seeds)
            assert list(prepared.matched_config(32).seeds)==list(prepared.matched_config(64).seeds)
            prepared.update_manifest(status='fresh_reproduction_'+args.phase)
            width.update(status='fresh_reproduction_'+args.phase)
            for d in policy['dimensions']:
                for seed in seeds:
                    if (run/'STOP_AFTER_PAIR').exists():
                        progress('stopped_at_requested_pair_boundary');return 3
                    progress('checking_external_training',dimension=d,seed=seed)
                    prepared.wait_for_external_training()
                    while width.external_gpu_job():
                        progress('waiting_for_external_training',dimension=d,seed=seed)
                        if (run/'STOP_AFTER_PAIR').exists():return 3
                        time.sleep(10)
                    progress('prepared_pair',dimension=d,seed=seed)
                    prepared.run_pair(d,seed,device,pilot=args.phase=='pilot')
                    original_provenance=load(archive/f'project/prepared_direct_comparison_20260921/data/d{d}_s{seed}/provenance.json')
                    fresh_provenance=load(prepared.OUT/f'data/d{d}_s{seed}/provenance.json')
                    assert original_provenance['feature_hashes']==fresh_provenance['feature_hashes'], ('Fresh data differ from archived data',d,seed)
                    progress('feature_width_pair',dimension=d,seed=seed)
                    width.run_pair(d,seed,device,pilot=args.phase=='pilot')
                    gc.collect();torch.cuda.empty_cache()
                    progress('pair_complete',dimension=d,seed=seed)
                    print(stamp(),'PAIR_FINISHED',d,seed,flush=True)
        complete=args.phase in ['full','analyze']
        progress('regenerating_summaries')
        import analyze as prepared_analysis
        prepared_analysis.analyze(require_complete=complete)
        width.analyze(complete=complete)
        if complete:
            subprocess.run([sys.executable,'-B',str(width.HERE/'analyze_new_pair.py')],check=True)
            prepared.update_manifest(status='fresh_reproduction_complete',completed_utc=stamp())
        else:
            prepared.update_manifest(status='fresh_reproduction_pilot_complete')
            width.update(status='fresh_reproduction_pilot_complete')
        progress('comparing_with_archive')
        comparison_out=run/('full_comparison' if complete else 'pilot_comparison')
        cmd=[sys.executable,'-B',str(args.comparator.resolve()),'--archive',str(archive),'--project',str(project),'--out',str(comparison_out),'--policy',str(args.policy.resolve()),'--stage','full' if complete else 'pilot']
        proc=subprocess.run(cmd,capture_output=True,text=True)
        (run/f'{args.phase}_comparison.stdout.txt').write_text(proc.stdout,encoding='utf-8')
        (run/f'{args.phase}_comparison.stderr.txt').write_text(proc.stderr,encoding='utf-8')
        print(proc.stdout,flush=True)
        if proc.stderr:print(proc.stderr,file=sys.stderr,flush=True)
        assert proc.returncode in (0,2),('Comparison driver failed',proc.returncode)
        result=load(comparison_out/'summary.json')
        progress('passed' if result['pass'] else 'requires_review',comparison_summary=str(comparison_out/'summary.json'))
        write(run/f'{args.phase}_result.json',{'finished_utc':stamp(),'elapsed_seconds':time.perf_counter()-started,'comparison_pass':result['pass'],'models_checked':result['completed_models'],'comparison_summary':str(comparison_out/'summary.json')})
        return 0 if result['pass'] else 2
    except BaseException as error:
        write(run/f'{args.phase}_failure.json',{'utc':stamp(),'error':str(error),'traceback':traceback.format_exc()})
        progress('failed',error=str(error))
        raise
    finally:
        for path in held_locks:
            if path.exists():os.replace(path,path.with_name(path.name+f'.ended_{args.phase}_{int(time.time())}'))

if __name__=='__main__':raise SystemExit(main())
