"""Explicitly lineage a newly authorized bounded experiment from an intact parent."""
import argparse
import json
import os
from pathlib import Path

from rvc_training import ROOT,WORKSPACE,verify_cache,checkpoint_set,child,disk_guard
from download_rvc_assets import sha256
from speech.diagnostics import atomic_json


def parent_checkpoint(workspace,current_identity):
    marker=json.loads((Path(workspace)/'recovery.json').read_text())
    source=Path(marker['parent_workspace']).resolve()
    parent=checkpoint_set(source)
    if not parent or parent[1]['sha256'] != marker['parent_sha256'] or parent[1]['identity'] != marker['parent_identity']:
        raise ValueError('Recovery parent checkpoint changed')
    for key in ('features_sha256','training_assets','source_commit'):
        if current_identity[key] != parent[1]['identity'][key]:
            raise ValueError('Recovery immutable identity mismatch: '+key)
    for key in ('schema','version','sample_rate','f0','batch_size','segment_size','seed'):
        if current_identity['config'][key] != parent[1]['identity']['config'][key]:
            raise ValueError('Recovery architecture/seed mismatch: '+key)
    if marker['authorization'] != 'User requested numerical recovery and Suno stem demonstration':
        raise ValueError('Missing recovery authorization record')
    return parent


def prepare(source,destination,mode):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if destination.exists():
        raise ValueError('Preserve existing experiment; refusing to reset its budget')
    if destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Recovery experiment must be separate from parent workspace')
    if mode not in ('diagnostic','recovery'):
        raise ValueError('Invalid recovery experiment mode')
    cache=verify_cache(source); parent=checkpoint_set(source)
    if not parent: raise ValueError('No complete parent checkpoint')
    disk_guard(ROOT,6*1024**3)
    destination.mkdir(parents=True)
    import shutil
    for clip in cache['clips']:
        original=child(source,clip['file']); target=child(destination,clip['file'])
        target.parent.mkdir(parents=True,exist_ok=True)
        try: os.link(original,target)
        except OSError: shutil.copyfile(original,target)
    # Both are immutable array inputs. No worker writes through these hard links.
    (destination/'features.json').write_bytes((source/'features.json').read_bytes())
    config=json.loads((ROOT/'configs/rvc_pilot.json').read_text())
    config.update(diagnostic=True,precision='fp16' if mode == 'diagnostic' else 'bf16')
    if mode == 'diagnostic': config.update(max_steps=80,max_seconds=300)
    atomic_json(destination/'config.json',config)
    atomic_json(destination/'recovery.json',dict(schema=1,mode=mode,parent_workspace=str(source),
                parent_sha256=parent[1]['sha256'],parent_identity=parent[1]['identity'],
                authorization='User requested numerical recovery and Suno stem demonstration',
                parent_budget_sha256=sha256(source/'budget.json'),original_failure='Nonfinite discriminator loss'))
    verify_cache(destination)
    return destination


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,default=WORKSPACE)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--mode',choices=('diagnostic','recovery'),required=True)
    args=parser.parse_args()
    print(prepare(args.parent,args.destination,args.mode))
