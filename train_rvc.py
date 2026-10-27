"""Bounded pilot controller: no model imports, shared GPU lock and durable budgets."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from rvc_training import (ROOT,WORKSPACE,CONFIG,load_config,verify_training_assets,verify_cache,
                          reserve_budget,reconcile_budget,checkpoint_set,disk_guard)
from download_rvc_assets import sha256
from speech.diagnostics import atomic_json
from speech.processes import ProcessTree


def gpu_memory():
    try:
        result = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],
                                         text=True,timeout=5,creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        return int(result.strip().splitlines()[0])
    except (OSError,ValueError,subprocess.SubprocessError):
        return None


def unresolved_failure(budget):
    attempts=budget.get('attempts',[])
    failures=[(i,a) for i,a in enumerate(attempts) if a.get('error')]
    if len(failures) == 1:
        index,failure=failures[0]
        if failure['stage'] == 'smoke' and 'out of memory' in failure['error'].lower():
            if any(a['stage'] == 'retry' and not a.get('error') for a in attempts[index+1:]):
                return False
    return bool(failures)


def execute(workspace=WORKSPACE,config_path=CONFIG,stage='smoke'):
    from rvc_runtime import verify_source,verify_assets,manifest,python_path
    from gpu_lock import gpu_job
    if stage not in ('smoke','retry','resume','pilot'):
        raise ValueError('Unknown pilot stage')
    workspace,config_path=Path(workspace).resolve(),Path(config_path).resolve()
    config=load_config(config_path)
    # Serialize preflight, budget reservation and worker lifetime across controllers.
    with gpu_job():
        verify_source(); verify_assets(); verify_training_assets(); verify_cache(workspace)
        checkpoint=checkpoint_set(workspace)
        identity=dict(features_sha256=sha256(workspace/'features.json'),config=config,
                      training_assets=manifest()['training_assets'],worker_sha256=sha256(ROOT/'rvc_training_worker.py'),
                      source_commit=manifest()['commit'])
        if checkpoint and checkpoint[1]['identity'] != identity:
            raise ValueError('Resume identity mismatch; refusing pretrained fallback')
        budget=json.loads((workspace/'budget.json').read_text()) if (workspace/'budget.json').exists() else dict(attempted_steps=0,seconds=0.)
        failed=[a for a in budget.get('attempts',[]) if a.get('error')]
        if unresolved_failure(budget) and not (stage == 'retry' and len(failed) == 1 and 'out of memory' in failed[0]['error'].lower()):
            raise RuntimeError('Prior training safety gate failed; further training needs a new reviewed plan')
        if stage == 'resume' and not checkpoint:
            raise ValueError('Resume verification requires a complete checkpoint')
        if stage == 'smoke' and budget['attempted_steps']:
            raise ValueError('Smoke was already attempted; use explicit bounded resume/pilot')
        if stage == 'retry':
            attempts=budget.get('attempts',[])
            if (checkpoint or config['segment_size'] != 8640 or len(attempts) != 1 or
                attempts[0]['stage'] != 'smoke' or 'out of memory' not in (attempts[0].get('error') or '').lower()):
                raise ValueError('One reduced-crop retry is allowed only after the first smoke OOM without a checkpoint')
        remaining_steps=config['max_steps']-budget['attempted_steps']
        remaining_seconds=config['max_seconds']-budget['seconds']
        smoke=stage in ('smoke','retry')
        steps=min(config['smoke_steps'] if smoke else 1 if stage == 'resume' else remaining_steps,remaining_steps)
        seconds=min(config['smoke_seconds'] if smoke else remaining_seconds,remaining_seconds)
        if seconds <= config['grace_seconds'] or steps <= 0:
            raise ValueError('Aggregate pilot budget exhausted')
        if stage == 'pilot' and not checkpoint:
            raise ValueError('Pilot requires a passing smoke checkpoint')
        if stage == 'pilot' and not any(a['stage'] == 'resume' and not a.get('error') for a in budget.get('attempts',[])):
            raise ValueError('Pilot requires successful resume verification first')
        disk_guard(workspace,3*1024**3)
        directory=workspace/('run-'+stage+'-'+uuid.uuid4().hex[:8]); directory.mkdir()
        atomic_json(directory/'request.json',dict(workspace=str(workspace),config=str(config_path),identity=identity,steps=steps,stage=stage))
        ticket=reserve_budget(workspace,config,steps,seconds)
        process=tree=None
        started=time.monotonic(); baseline=gpu_memory(); peak=baseline; samples=[]
        error=None; last_progress=started; last_key=None; next_sample=started
        try:
            env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
            flags=(subprocess.CREATE_NO_WINDOW|0x00000004) if os.name == 'nt' else 0
            with (directory/'worker.log').open('wb') as log:
                process=subprocess.Popen([str(python_path()),'-X','utf8','-B',str(ROOT/'rvc_training_worker.py'),str(directory)],
                                         cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                         creationflags=flags,start_new_session=os.name != 'nt')
                try:
                    tree=ProcessTree(process); tree.resume()
                except BaseException:
                    process.kill(); process.wait(timeout=10); raise
                (directory/'go').touch()
                while True:
                    now=time.monotonic(); elapsed=now-started
                    if process.poll() is not None:
                        if process.returncode:
                            details=json.loads((directory/'error.json').read_text()) if (directory/'error.json').exists() else {}
                            raise RuntimeError(details.get('message',f'Training worker exited {process.returncode}'))
                        break
                    if elapsed >= seconds:
                        raise TimeoutError('Pilot worker hard deadline reached')
                    if elapsed >= seconds-config['grace_seconds']:
                        (directory/'stop').touch()
                    try:
                        state=json.loads((directory/'state.json').read_text())
                        key=(state.get('stage'),state.get('chunk'),state.get('step'))
                        if key != last_key: last_key,last_progress=key,now
                    except (OSError,ValueError): pass
                    if now-last_progress > config['step_timeout']:
                        raise TimeoutError('Training progress heartbeat timed out')
                    if os.fstat(log.fileno()).st_size > 8*1024**2:
                        raise RuntimeError('Training diagnostic log limit exceeded')
                    if now >= next_sample:
                        used=gpu_memory(); next_sample=now+2
                        if used is not None:
                            peak=max(peak or 0,used); samples.append([round(elapsed,2),used])
                            if used >= 4096:
                                raise RuntimeError('Device-wide VRAM ceiling reached')
                    time.sleep(.1)
                result=json.loads((directory/'training.json').read_text())
                if result['successful_steps'] < 1:
                    raise RuntimeError('No successful paired optimizer updates')
        except BaseException as exc:
            error=str(exc)
            raise
        finally:
            if tree: tree.close()
            elapsed=time.monotonic()-started
            # Reconcile only after containment has confirmed process-tree death.
            progress=json.loads((directory/'progress.json').read_text()) if (directory/'progress.json').exists() else dict(attempted_steps=0)
            reconcile_budget(workspace,ticket,progress['attempted_steps'],elapsed,dict(stage=stage,error=error,directory=directory.name))
            recovery=gpu_memory()
            record=dict(pid=process.pid if process else None,exited=process is not None and process.poll() is not None,
                        returncode=process.returncode if process else None,elapsed_seconds=elapsed,
                        baseline_gpu_mib=baseline,peak_device_gpu_mib=peak,after_gpu_mib=recovery,samples=samples,error=error)
            atomic_json(directory/'process.json',record)
        print(json.dumps(dict(directory=str(directory),**result),indent=2))
        return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=WORKSPACE)
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--stage',choices=('smoke','retry','resume','pilot'),default='smoke')
    args=parser.parse_args()
    execute(args.workspace,args.config,args.stage)
