"""Private bounded RVC worker; receives local JSON and produces PCM24 only."""
import json
from pathlib import Path
import sys
import time


def main():
    directory = Path(sys.argv[1]).resolve()
    deadline = time.monotonic()+15
    while not (directory/'go').exists():
        if time.monotonic() >= deadline:
            return 2
        time.sleep(.02)
    from speech.diagnostics import Trace,atomic_json,fingerprint
    trace = Trace(directory)
    trace.event('worker_start')
    try:
        import numpy as np
        import soundfile as sf
        import torch
        from audio_io import resample,write_wav,validate_audio
        from rvc_audio import segment_ranges,fit_segment,stitch_segments,validate_conversion
        from rvc_runtime import load_models,verify_source,verify_assets
        request = json.loads((directory/'request.json').read_text(encoding='utf-8'))
        options = request['options']
        verify_source(); verify_assets()
        trace.event('checkpoint_provenance',checkpoint=fingerprint(options['model']),
                    index=fingerprint(options['index']) if options['index_rate'] else None)
        hubert,net,pipeline,metadata = load_models(options,trace)
        native = metadata['sample_rate']
        parts = []
        with sf.SoundFile(request['input']) as source:
            if (source.samplerate,source.subtype,source.channels) != (48000,'PCM_24',1):
                raise ValueError('Input audio contract changed before worker read')
            frames = len(source)
            if not .15 <= frames/48000 <= options['max_input_seconds']:
                raise ValueError('Input duration is outside the configured limit')
            for i,(start,end,left,right) in enumerate(segment_ranges(frames,48000)):
                trace.begin_chunk(i)
                source.seek(left)
                samples = validate_audio(source.read(right-left,dtype='float32'),48000)
                features = resample(samples,48000,16000)
                # A very short tail still receives real context; pad only internal model input.
                if len(features) < 3200:
                    features = np.pad(features,(0,3200-len(features)),mode='reflect')
                with torch.inference_mode():
                    raw = pipeline.pipeline(hubert,net,options['speaker_id'],features,
                        f"{request['input']}:{start}",[0,0,0],options['transpose'],
                        options['f0_predictor'],options['index'] or '',options['index_rate'],
                        metadata['f0'],3,native,0,1.0,metadata['version'],.33)
                raw = validate_audio(raw,native)
                peak = float(np.max(np.abs(raw)))
                if peak > 8:
                    raise ValueError('Implausible decoder output amplitude')
                offset = round((start-left)*native/48000)
                needed = round(end*native/48000)-round(start*native/48000)
                core = fit_segment(raw[offset:offset+needed],needed,native)
                parts.append((round(start*native/48000),core))
                trace.event('segment_converted',start_sample=start,end_sample=end,native_peak=peak)
        trace.phase = 'finalize'
        trace.event('assembly_start')
        native_audio = stitch_segments(parts,native,round(frames*native/48000))
        metrics = validate_conversion(native_audio,native,frames/48000)
        final = fit_segment(resample(native_audio,native,48000),frames,48000)
        validate_conversion(final,48000,frames/48000)
        write_wav(directory/'result.wav',final,48000)
        atomic_json(directory/'result.json',dict(metrics=metrics,segments=len(parts),native_sample_rate=native))
        trace.event('complete',**metrics)
        return 0
    except Exception as exc:
        import traceback
        trace.event('failed',error=str(exc),traceback=traceback.format_exc())
        atomic_json(directory/'error.json',dict(type=type(exc).__name__,message=str(exc)))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
