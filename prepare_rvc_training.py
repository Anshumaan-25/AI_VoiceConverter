"""Prepare split-isolated content/pitch arrays without writing resampled audio."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from rvc_training import (DATASET,WORKSPACE,child,dataset_manifest,extraction_identity,
                          aligned_frames,disk_guard,verify_cache)
from speech.diagnostics import atomic_json,Trace


def worker(directory):
    while not (directory/'go').exists():
        time.sleep(.02)
    trace = Trace(directory); trace.event('worker_start')
    try:
        import numpy as np
        import soundfile as sf
        import torch
        from audio_io import resample
        from rvc_runtime import SOURCE,manifest,verify_source,verify_assets
        verify_source(); verify_assets()
        sys.path.insert(0,str(SOURCE))
        request = json.loads((directory/'request.json').read_text())
        workspace,dataset = Path(request['workspace']),Path(request['dataset'])
        phase = request['phase']
        data,_ = dataset_manifest(dataset)
        if request['identity'] != extraction_identity(dataset,data):
            raise ValueError('Extraction identity changed')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; extraction requires the selected GPU')
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(min(.85,3.4*1024**3/torch.cuda.get_device_properties(0).total_memory))
        if phase == 'content':
            from fairseq import checkpoint_utils
            models,cfg,_ = checkpoint_utils.load_model_ensemble_and_task([str(SOURCE/manifest()['assets']['hubert']['file'])],suffix='')
            model = models[0].eval().cuda().half()
        else:
            from infer.lib.rmvpe import RMVPE
            model = RMVPE(str(SOURCE/manifest()['assets']['rmvpe']['file']),is_half=True,device='cuda:0')
        trace.event('extractor_loaded',phase_name=phase)
        results=[]
        for i,clip in enumerate(data['clips']):
            trace.begin_chunk(i)
            audio,sr = sf.read(str(child(dataset,clip['file'])),dtype='float32')
            info = sf.info(str(child(dataset,clip['file'])))
            if (sr,info.subtype,info.channels) != (48000,'PCM_24',1) or not np.isfinite(audio).all():
                raise ValueError('Training audio format/scaling invalid')
            audio16 = resample(audio,48000,16000)
            target = workspace/'arrays'/clip['split']/(Path(clip['file']).stem+'.'+phase+'.npy')
            target.parent.mkdir(parents=True,exist_ok=True)
            with torch.inference_mode():
                if phase == 'content':
                    tensor = torch.from_numpy(audio16).cuda().half()[None]
                    if cfg.task.normalize:
                        tensor = torch.nn.functional.layer_norm(tensor,tensor.shape[1:])
                    value = model.extract_features(source=tensor,padding_mask=torch.zeros(tensor.shape,dtype=torch.bool,device='cuda:0'),output_layer=12)[0][0].float().cpu().numpy()
                    del tensor
                else:
                    value = model.infer_from_audio(audio16,thred=.03).astype(np.float32)
            if not np.isfinite(value).all() or (phase == 'content' and (value.ndim != 2 or value.shape[1] != 768)):
                raise ValueError('Invalid extracted tensor')
            temporary = target.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.save(stream,value,allow_pickle=False)
            os.replace(temporary,target)
            results.append(dict(source_file=clip['file'],source_sha256=clip['sha256'],split=clip['split'],
                                file=str(target.relative_to(workspace)),audio_frames=len(audio),
                                shape=list(value.shape)))
            trace.event('clip_complete',frames=len(value))
            del value,audio,audio16
        atomic_json(directory/'extraction.json',dict(clips=results,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        trace.event('complete')
        return 0
    except Exception as exc:
        import traceback
        atomic_json(directory/'error.json',dict(message=str(exc),traceback=traceback.format_exc()))
        return 1


def prepare(dataset=DATASET,workspace=WORKSPACE):
    import numpy as np
    from rvc_runtime import ROOT,python_path,verify_source,verify_assets
    from gpu_lock import gpu_job
    from speech.contracts import SpeechOptions
    from speech.supervisor import run_worker
    from download_rvc_assets import sha256
    workspace,dataset = Path(workspace).resolve(),Path(dataset).resolve()
    if workspace == dataset or workspace.is_relative_to(dataset) or dataset.is_relative_to(workspace):
        raise ValueError('Training workspace and immutable dataset must be separate')
    verify_source(); verify_assets()
    data,counts = dataset_manifest(dataset)
    identity = extraction_identity(dataset,data)
    if (workspace/'features.json').exists():
        return verify_cache(workspace)
    if workspace.exists() and any(workspace.iterdir()):
        raise ValueError('Incomplete training workspace; preserve it and select a new path')
    disk_guard(ROOT,6*1024**3)  # arrays + two resume sets + checkpoint staging + export
    workspace.mkdir(parents=True,exist_ok=True)
    atomic_json(workspace/'identity.json',identity)
    phases={}
    for phase in ('content','pitch'):
        directory = workspace/phase; directory.mkdir()
        atomic_json(directory/'request.json',dict(workspace=str(workspace),dataset=str(dataset),identity=identity,phase=phase))
        with gpu_job():
            run_worker([str(python_path()),'-X','utf8','-B',str(Path(__file__).resolve()),'--worker',str(directory)],
                       directory,SpeechOptions(load_timeout=180,chunk_timeout=120),time.monotonic()+1800,backend='bark')
        phases[phase] = json.loads((directory/'extraction.json').read_text())
    clips=[]
    for content,pitch in zip(phases['content']['clips'],phases['pitch']['clips']):
        if content['source_file'] != pitch['source_file']:
            raise ValueError('Extraction order mismatch')
        frames = aligned_frames(content['audio_frames'],content['shape'][0],pitch['shape'][0])
        features = np.load(child(workspace,content['file']),allow_pickle=False)
        f0 = np.load(child(workspace,pitch['file']),allow_pickle=False)[:frames]
        mel = 1127*np.log1p(f0/700)
        coarse = np.rint(np.clip((mel-1127*np.log1p(50/700))*254/(1127*np.log1p(1100/700)-1127*np.log1p(50/700))+1,1,255)).astype(np.int64)
        target = workspace/'arrays'/content['split']/(Path(content['source_file']).stem+'.npz')
        np.savez(target,features=np.repeat(features,2,axis=0)[:frames],pitch=f0,coarse=coarse)
        clips.append(dict(source_file=content['source_file'],source_sha256=content['source_sha256'],split=content['split'],
                          file=str(target.relative_to(workspace)),sha256=sha256(target),frames=frames,
                          source_frames=content['audio_frames'],tail_samples_excluded=content['audio_frames']-frames*480))
    result=dict(status='complete',dataset=str(dataset),identity=identity,counts=counts,clips=clips,
                exclusions=[],alignment_tolerance_frames=4,validation_scope='same-session temporal holdout')
    atomic_json(workspace/'features.json',result)
    return verify_cache(workspace)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=DATASET)
    parser.add_argument('--workspace',type=Path,default=WORKSPACE)
    parser.add_argument('--worker',type=Path)
    args=parser.parse_args()
    if args.worker:
        raise SystemExit(worker(args.worker))
    result=prepare(args.dataset,args.workspace)
    print(json.dumps(dict(counts=result['counts'],clips=len(result['clips']),workspace=str(args.workspace)),indent=2))
