"""Contained Demucs separation; native-rate tensors stay in memory, WAVs are PCM24/48k."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import uuid

from speech.diagnostics import Trace,atomic_json
from download_rvc_assets import sha256


def select_interval(vocal,rate,seconds=30):
    """Select an early sustained-vocal window; energy is not proof of stem purity."""
    import numpy as np
    vocal=np.asarray(vocal,dtype=np.float32)
    if type(rate) is not int or rate <= 0 or type(seconds) is not int or not 20 <= seconds <= 30:
        raise ValueError('Invalid sample rate or excerpt duration')
    if vocal.ndim not in (1,2) or not vocal.size or not np.isfinite(vocal).all():
        raise ValueError('Invalid vocal audio')
    if vocal.ndim == 2: vocal=vocal.mean(axis=1)
    count=len(vocal)//rate
    if count < seconds: raise ValueError('Track is shorter than the requested excerpt')
    rms=np.array([np.sqrt(np.mean(vocal[i*rate:(i+1)*rate].astype(float)**2)) for i in range(count)])
    threshold=max(.005,float(np.quantile(rms,.75))*.2)
    candidates=[]
    for start in range(count-seconds+1):
        window=rms[start:start+seconds]
        fraction=float(np.mean(window>threshold))
        candidates.append((fraction,float(np.median(window)),start))
    sustained=[x for x in candidates if x[0] >= .95 and x[1] >= threshold*2]
    chosen=sustained[0] if sustained else max(candidates,key=lambda x:(x[0],x[1],-x[2]))
    if chosen[0] < .8 or chosen[1] <= threshold:
        raise ValueError('No sufficiently active vocal excerpt was found')
    return dict(start_seconds=max(0.,chosen[2]-.2),seconds=seconds,active_second_fraction=chosen[0],
                median_vocal_rms=chosen[1],energy_threshold=threshold,
                method='Earliest sustained-vocal window with 200 ms leading context; energy-based, listening pending')


def worker(directory):
    while not (directory/'go').exists(): time.sleep(.02)
    trace=Trace(directory); trace.event('worker_start')
    try:
        import importlib.metadata as packages
        import urllib.parse
        import numpy as np
        import torch
        import demucs
        from demucs.pretrained import get_model,_parse_remote_files,REMOTE_ROOT
        from demucs.apply import apply_model
        from audio_io import read_audio,resample,write_wav
        request=json.loads((directory/'request.json').read_text())
        if packages.version('demucs') != '4.0.1' or torch.__version__ != '2.1.0+cu121':
            raise RuntimeError('Expected the inspected Demucs 4.0.1 / torch 2.1.0+cu121 runtime')
        torch.set_num_threads(4); torch.manual_seed(42)
        device=request['device']
        if device == 'cuda':
            if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
            torch.cuda.set_per_process_memory_fraction(min(.85,3.4*1024**3/torch.cuda.get_device_properties(0).total_memory))
        # All four assets must already exist; no unrecorded model download is needed.
        urls=_parse_remote_files(REMOTE_ROOT/'files.txt')
        assets=[]
        for signature in ('f7e0c4bc','d12395a8','92cfc3b6','04573f0d'):
            name=Path(urllib.parse.urlparse(urls[signature]).path).name
            path=Path(torch.hub.get_dir())/'checkpoints'/name
            expected_prefix=Path(name).stem.split('-',1)[1]
            digest=sha256(path)
            if not digest.startswith(expected_prefix): raise ValueError('Cached Demucs asset hash prefix mismatch: '+name)
            assets.append(dict(file=name,sha256=digest,expected_sha256_prefix=expected_prefix,source=urls[signature]))
        model=get_model('htdemucs_ft').cpu().eval()
        trace.event('demucs_loaded',sources=model.sources,device=device)
        audio,sr=read_audio(request['input'],model.samplerate,mono=False)
        if audio.shape[1] != 2 or not 30 <= len(audio)/sr <= 600:
            raise ValueError('Expected a stereo song between 30 seconds and ten minutes')
        mix=torch.from_numpy(audio.T.copy())
        reference=mix.mean(0); mean=reference.mean(); std=reference.std()
        if not torch.isfinite(mix).all() or not torch.isfinite(std) or std <= 1e-5:
            raise ValueError('Song contains invalid or silent audio')
        normalized=(mix-mean)/std
        counter=[0]
        def begin(mod,args):
            trace.begin_chunk(counter[0]); counter[0]+=1
        def complete(mod,args,result): trace.event('demucs_segment_complete')
        for submodel in model.models:
            submodel.register_forward_pre_hook(begin)
            submodel.register_forward_hook(complete)
        with torch.inference_mode():
            estimates=apply_model(model,normalized[None],device=device,shifts=0,split=True,
                                  overlap=.5,progress=False,num_workers=0)[0]
            estimates=estimates*std+mean
        if not torch.isfinite(estimates).all(): raise ValueError('Nonfinite separated audio')
        trace.phase='finalize'; trace.event('separation_complete')
        index=model.sources.index('vocals')
        vocals=estimates[index].T.cpu().numpy()
        instrumental=sum(estimates[i] for i in range(len(model.sources)) if i != index).T.cpu().numpy()
        vocals=resample(vocals,sr,48000); instrumental=resample(instrumental,sr,48000)
        peak=max(float(np.max(np.abs(vocals))),float(np.max(np.abs(instrumental))))
        common_gain=min(1.,.99/peak)
        vocals*=common_gain; instrumental*=common_gain
        destination=Path(request['destination'])
        write_wav(destination/'vocals.wav',vocals,48000)
        write_wav(destination/'instrumental.wav',instrumental,48000)
        selection=select_interval(vocals,48000)
        first=round(selection['start_seconds']*48000); last=first+1440000
        write_wav(destination/'vocals-30s.wav',vocals[first:last].mean(axis=1),48000)
        write_wav(destination/'instrumental-30s.wav',instrumental[first:last],48000)
        if sha256(request['input']) != request['input_sha256']: raise ValueError('Source changed during separation')
        package=Path(demucs.__file__).parent
        report=dict(status='complete',input=request['input'],input_sha256=request['input_sha256'],model='htdemucs_ft',
                    demucs=packages.version('demucs'),torch=torch.__version__,device=device,native_sample_rate=sr,
                    persisted_sample_rate=48000,persisted_subtype='PCM_24',source_frames=len(audio),output_frames=len(vocals),
                    overlap=.5,shifts=0,num_workers=0,common_stem_gain=common_gain,selection=selection,assets=assets,
                    source_code_sha256={name:sha256(package/name) for name in ('apply.py','htdemucs.py','pretrained.py')},
                    peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
                    peak_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else 0,
                    outputs={p.name:sha256(p) for p in destination.glob('*.wav')},
                    limitation='Separation may retain bleed or metallic artifacts; listening is needed to judge cleanliness')
        atomic_json(destination/'separation.json',report); trace.event('complete')
        return 0
    except Exception as exc:
        import traceback
        atomic_json(directory/'error.json',dict(message=str(exc),type=type(exc).__name__,traceback=traceback.format_exc()))
        return 1


def separate(source,python,destination,device='cuda'):
    from gpu_lock import gpu_job
    from contextlib import nullcontext
    from speech.contracts import SpeechOptions
    from speech.supervisor import run_worker
    import soundfile as sf
    source,python,destination=map(lambda p:Path(p).resolve(),(source,python,destination))
    if not source.is_file() or not python.is_file(): raise FileNotFoundError('Input song or Demucs Python not found')
    info=sf.info(str(source))
    if info.channels != 2 or not 30 <= info.duration <= 600:
        raise ValueError('Expected a stereo song between 30 seconds and ten minutes')
    if destination.exists(): raise ValueError('Preserve previous separation; choose a new destination')
    if source.is_relative_to(destination): raise ValueError('Separation destination overlaps input')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(destination.parent).free < 10*1024**3+512*1024**2:
        raise RuntimeError('Separation would violate the free-space reserve')
    staged=destination.with_name(destination.name+'.staging-'+uuid.uuid4().hex[:8]); staged.mkdir()
    directory=staged/'worker'; directory.mkdir()
    atomic_json(directory/'request.json',dict(input=str(source),input_sha256=sha256(source),
                destination=str(staged),device=device))
    with gpu_job() if device=='cuda' else nullcontext():
        run_worker([str(python),'-X','utf8','-B',str(Path(__file__).resolve()),'--worker',str(directory)],
                   directory,SpeechOptions(device=device,load_timeout=180,chunk_timeout=120),
                   time.monotonic()+1800,backend='bark' if device=='cuda' else 'cpu')
    staged.rename(destination)
    return destination


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path)
    parser.add_argument('--python',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--worker',type=Path)
    args=parser.parse_args()
    if args.worker: raise SystemExit(worker(args.worker))
    if not all((args.input,args.python,args.output_dir)): parser.error('--input, --python and --output-dir are required')
    print(separate(args.input,args.python,args.output_dir,args.device))
