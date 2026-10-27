"""Controller-safe validation, provenance and atomic pilot checkpoint helpers."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

from download_rvc_assets import sha256
from rvc_runtime import ROOT, SOURCE, manifest
from speech.diagnostics import atomic_json

DATASET = ROOT/'datasets/rvc/anshumaan/0dddb82392f1860af8a3'
WORKSPACE = ROOT/'logs/milestone3/training'
CONFIG = ROOT/'configs/rvc_pilot.json'


def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def child(root, relative):
    root = Path(root).resolve()
    path = (root/relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Artifact path escapes its workspace')
    return path


def disk_guard(root, additional=0):
    if shutil.disk_usage(root).free < 10*1024**3+additional:
        raise RuntimeError('Operation would violate the 10 GiB free-space reserve')


def verify_training_assets():
    for asset in manifest()['training_assets'].values():
        path = SOURCE/asset['file']
        if not path.is_file() or path.stat().st_size != asset['bytes'] or sha256(path) != asset['sha256']:
            raise RuntimeError('Training asset missing or hash mismatch: '+path.name)


def load_config(path=CONFIG):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    expected = dict(schema=1,version='v2',sample_rate=48000,f0=1,batch_size=1)
    if any(config.get(k) != v for k,v in expected.items()):
        raise ValueError('Pilot requires v2, 48 kHz, pitch conditioning and batch size 1')
    limits = dict(max_steps=500,max_seconds=1800,smoke_steps=10,smoke_seconds=300,save_every=100,
                  step_timeout=120,grace_seconds=20,max_consecutive_skips=8)
    for name,maximum in limits.items():
        if type(config.get(name)) is not int or not 1 <= config[name] <= maximum:
            raise ValueError('Invalid bounded pilot setting: '+name)
    if config.get('segment_size') not in (17280,8640):
        raise ValueError('Only the original crop or one reduced crop is supported')
    if type(config.get('seed')) is not int:
        raise ValueError('Seed must be an integer')
    return config


def dataset_manifest(dataset):
    dataset = Path(dataset).resolve()
    data = json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    if data.get('status') != 'complete' or not data.get('clips'):
        raise ValueError('Dataset is incomplete')
    pcm_splits = {}
    counts = {'train':0,'val':0}
    for source in data['sources']:
        if sha256(source['path']) != source['sha256']:
            raise ValueError('Original source changed')
    for clip in data['clips']:
        path = child(dataset,clip['file'])
        if clip['split'] not in counts or sha256(path) != clip['sha256']:
            raise ValueError('Invalid split or altered dataset clip')
        if clip['pcm_sha256'] in pcm_splits and pcm_splits[clip['pcm_sha256']] != clip['split']:
            raise ValueError('Exact PCM duplicate across splits')
        pcm_splits[clip['pcm_sha256']] = clip['split']
        source = next(s for s in data['sources'] if s['sha256'] == clip['source'])
        intervals = [i for i in source['split_intervals'] if i['split'] == clip['split']]
        if not any(i['start_sample'] <= clip['start_sample'] < clip['end_sample'] <= i['end_sample'] for i in intervals):
            raise ValueError('Clip crosses a source split boundary')
        counts[clip['split']] += 1
    if not all(counts.values()):
        raise ValueError('Both training and provisional validation must be nonempty')
    return data,counts


def extraction_identity(dataset, data):
    return dict(schema=1,dataset_manifest_sha256=sha256(Path(dataset)/'manifest.json'),
                source_commit=manifest()['commit'],patch_sha256=sha256(ROOT/manifest()['patch']),
                assets=manifest()['assets'],sample_rate=48000,feature_width=768,layer=12,
                code={p:sha256(ROOT/p) for p in ('prepare_rvc_training.py','rvc_training.py')})


def verify_cache(workspace):
    workspace = Path(workspace).resolve()
    data = json.loads((workspace/'features.json').read_text(encoding='utf-8'))
    dataset = Path(data['dataset'])
    original,_ = dataset_manifest(dataset)
    if data['identity'] != extraction_identity(dataset,original):
        raise ValueError('Stale feature cache identity; create a new workspace')
    original_clips = {c['file']:c for c in original['clips']}
    if len(data['clips']) != len(original_clips):
        raise ValueError('Feature cache omits dataset clips')
    seen = set()
    for clip in data['clips']:
        source = original_clips.get(clip['source_file'])
        if source is None or clip['source_file'] in seen or source['sha256'] != clip['source_sha256'] or source['split'] != clip['split']:
            raise ValueError('Feature provenance/split mismatch')
        seen.add(clip['source_file'])
        if sha256(child(workspace,clip['file'])) != clip['sha256']:
            raise ValueError('Feature cache artifact changed')
    return data


def aligned_frames(audio_frames, feature_frames, pitch_frames, tolerance=4):
    expected = audio_frames//480
    sizes = (expected,feature_frames*2,pitch_frames)
    if min(sizes) < 1 or max(sizes)-min(sizes) > tolerance:
        raise ValueError(f'Material feature/pitch/audio alignment error: {sizes}')
    return min(sizes)


def reserve_budget(workspace, config, steps, seconds):
    path = Path(workspace)/'budget.json'
    budget = json.loads(path.read_text()) if path.exists() else dict(schema=1,attempted_steps=0,seconds=0.,attempts=[])
    if budget.get('active'):
        raise RuntimeError('Unreconciled prior worker reservation; inspect process evidence before any recovery')
    if steps < 1 or seconds <= 0 or budget['attempted_steps']+steps > config['max_steps'] or budget['seconds']+seconds > config['max_seconds']:
        raise ValueError('Aggregate pilot budget exhausted')
    ticket = dict(id=uuid.uuid4().hex,steps=steps,seconds=seconds)
    budget['active'] = ticket
    # Pessimistic reservation survives controller termination; never silently resets.
    budget['attempted_steps'] += steps
    budget['seconds'] += seconds
    atomic_json(path,budget)
    return ticket


def reconcile_budget(workspace,ticket,attempted,seconds,record):
    path = Path(workspace)/'budget.json'
    budget = json.loads(path.read_text())
    if budget['active'] != ticket or not 0 <= attempted <= ticket['steps']:
        raise ValueError('Invalid budget reconciliation')
    budget['attempted_steps'] += attempted-ticket['steps']
    budget['seconds'] += min(seconds,ticket['seconds'])-ticket['seconds']
    budget['active'] = None
    budget['attempts'].append(dict(ticket=ticket,attempted_steps=attempted,seconds=seconds,**record))
    atomic_json(path,budget)


def checkpoint_set(workspace):
    workspace = Path(workspace)
    pointer = workspace/'latest.json'
    if not pointer.exists():
        return None
    latest = json.loads(pointer.read_text())
    directory = child(workspace,latest['directory'])
    metadata = json.loads((directory/'manifest.json').read_text())
    if metadata.get('status') != 'complete' or metadata != latest['manifest']:
        raise ValueError('Partial or mismatched checkpoint set')
    state = child(directory,metadata['file'])
    if not state.is_file() or sha256(state) != metadata['sha256']:
        raise ValueError('Corrupt checkpoint set; refusing pretrained fallback')
    return state,metadata


def publish_checkpoint(workspace,step,identity,save):
    workspace = Path(workspace).resolve()
    disk_guard(workspace,2*1024**3)
    name = f'checkpoint-{step:06d}-{uuid.uuid4().hex[:8]}'
    directory = child(workspace,name+'.staging'); directory.mkdir()
    try:
        state = directory/'state.pth'
        save(state)
        metadata = dict(status='complete',file=state.name,step=step,identity=identity,
                        sha256=sha256(state),bytes=state.stat().st_size)
        atomic_json(directory/'manifest.json',metadata)
        published = child(workspace,name)
        os.replace(directory,published)
        atomic_json(workspace/'latest.json',dict(directory=name,manifest=metadata))
        # Verify new bytes before pruning only this module's known complete sets.
        checkpoint_set(workspace)
        complete = []
        for p in workspace.glob('checkpoint-*'):
            if p.is_dir() and not p.is_symlink() and not p.name.endswith('.staging'):
                m = p/'manifest.json'
                if m.is_file():
                    item = json.loads(m.read_text())
                    if item.get('status') == 'complete' and item.get('identity') == identity:
                        complete.append((p.stat().st_mtime_ns,p))
        for _,old in sorted(complete,reverse=True)[2:]:
            old = child(workspace,old.name)
            shutil.rmtree(old)
        return metadata
    except BaseException:
        # Interrupted staging is diagnostic evidence; it is never a resume candidate.
        raise
