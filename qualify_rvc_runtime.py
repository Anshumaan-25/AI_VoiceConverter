"""Real HuBERT/RMVPE GPU probe. This does not qualify a trained voice converter."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent


def worker(directory):
    while not (directory/'go').exists():
        time.sleep(.02)
    from speech.diagnostics import Trace,atomic_json
    trace = Trace(directory)
    trace.event('worker_start')
    try:
        import numpy as np
        import torch
        from audio_io import read_audio
        from rvc_runtime import SOURCE,manifest,verify_source,verify_assets
        verify_source(); verify_assets()
        sys.path.insert(0,str(SOURCE))
        from fairseq import checkpoint_utils
        from infer.lib.rmvpe import RMVPE
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable for the explicit asset probe')
        torch.set_num_threads(4)
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(.85,3.4*1024**3/total))
        models,_,_ = checkpoint_utils.load_model_ensemble_and_task(
            [str(SOURCE/manifest()['assets']['hubert']['file'])],suffix='')
        hubert = models[0].eval().to('cuda:0').half()
        pitch_model = RMVPE(str(SOURCE/manifest()['assets']['rmvpe']['file']),is_half=True,device='cuda:0')
        trace.event('assets_loaded')
        trace.begin_chunk(0)
        request = json.loads((directory/'request.json').read_text())
        audio,sr = read_audio(request['audio'],16000)
        if not .15 <= len(audio)/sr <= 5:
            raise ValueError('Asset probe expects 0.15–5 seconds of speech')
        with torch.inference_mode():
            tensor = torch.from_numpy(audio).to('cuda:0').half()[None]
            features = hubert.extract_features(source=tensor,
                padding_mask=torch.zeros(tensor.shape,dtype=torch.bool,device='cuda:0'),output_layer=12)[0]
            f0 = pitch_model.infer_from_audio(audio,thred=.03)
            torch.cuda.synchronize()
        if features.shape[-1] != 768 or not torch.isfinite(features).all() or not np.isfinite(f0).all():
            raise RuntimeError('Invalid real encoder/pitch output')
        metrics = dict(scope='encoder_and_pitch_only_not_voice_conversion',feature_shape=list(features.shape),
                       pitch_frames=len(f0),voiced_frames=int(np.count_nonzero(f0)),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved())
        atomic_json(directory/'probe.json',metrics)
        trace.event('complete',**metrics)
        return 0
    except Exception as exc:
        import traceback
        atomic_json(directory/'error.json',dict(message=str(exc),traceback=traceback.format_exc()))
        return 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--worker',type=Path)
    args=parser.parse_args()
    if args.worker:
        return worker(args.worker)
    if not args.audio or not args.output_dir:
        parser.error('--audio and --output-dir are required')
    from gpu_lock import gpu_job
    from speech.contracts import SpeechOptions
    from speech.diagnostics import atomic_json
    from speech.supervisor import run_worker
    from rvc_runtime import python_path
    args.output_dir.mkdir(parents=True,exist_ok=False)
    atomic_json(args.output_dir/'request.json',dict(audio=str(args.audio.resolve())))
    with gpu_job():
        run_worker([str(python_path()),'-X','utf8','-B',str(Path(__file__).resolve()),
                    '--worker',str(args.output_dir.resolve())],args.output_dir,SpeechOptions(),
                   time.monotonic()+300,backend='bark')
    print((args.output_dir/'probe.json').read_text())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
