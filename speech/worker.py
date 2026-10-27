"""Private worker entry point. Wait for job containment before importing models."""
import json
import os
from pathlib import Path
import sys
import time


def main():
    directory = Path(sys.argv[1]).resolve()
    deadline = time.monotonic()+15
    while not (directory/'go').exists():
        if time.monotonic() > deadline:
            return 2
        time.sleep(.02)
    from .diagnostics import Trace, atomic_json
    trace = Trace(directory)
    trace.event('worker_start')
    try:
        import importlib.metadata
        import numpy as np
        from audio_io import write_wav, resample
        from .contracts import SpeechOptions
        from .backends import load_backend
        from .text import duration_budget
        from .validation import SpeechValidator, finish_chunk
        request = json.loads((directory/'request.json').read_text(encoding='utf-8'))
        options = SpeechOptions(**request['options'])
        options.validate()
        versions = {}
        for name in ('numpy','soundfile','webrtcvad-wheels','torch','torchaudio','suno-bark','TTS','encodec','transformers'):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        trace.event('runtime', python=sys.version, versions=versions)
        backend = load_backend(request['backend'], options, trace)
        validator = SpeechValidator()
        reports = []
        parts = []
        for index, text in enumerate(request['chunks']):
            trace.begin_chunk(index)
            budget = duration_budget(text, options.max_chunk_seconds)
            raw, sr = backend.generate(text, budget, index)
            trace.event('raw_audio', frames=len(raw), sample_rate=sr,
                        peak=float(np.max(np.abs(raw))) if len(raw) else 0,
                        note='Diagnostic WAV attenuates peaks above 0.99; raw metrics precede attenuation')
            if options.diagnostics:
                write_wav(directory/f'{index:03d}-raw.wav', raw, sr)
            trace.event('validation_start', native_sample_rate=sr)
            report = validator.validate(raw, sr, text, budget)
            clean = finish_chunk(raw, sr)
            clean = resample(clean, sr, 48000)
            validator.validate(clean, 48000, text, budget)
            write_wav(directory/f'{index:03d}-validated.wav', clean, 48000)
            report.update(index=index, text=text, native_sample_rate=sr,
                          output_frames=len(clean), duration_limit=budget)
            reports.append(report)
            parts.append(clean)
            trace.event('chunk_validated', **report)
        trace.phase = 'finalize'
        trace.event('stitch_start')
        stitched = []
        for i, part in enumerate(parts):
            if i:
                stitched.append(np.zeros(7200, dtype=np.float32))
            stitched.append(part)
        write_wav(directory/'result.wav', np.concatenate(stitched), 48000)
        atomic_json(directory/'result.json', dict(backend=request['backend'], chunks=reports))
        trace.event('complete')
        return 0
    except Exception as exc:
        import traceback
        trace.event('failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        atomic_json(directory/'error.json', dict(type=type(exc).__name__, message=str(exc)))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
