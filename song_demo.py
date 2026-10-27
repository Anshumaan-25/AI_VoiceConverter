"""Convert a synchronized 30-second vocal stem and mix its original accompaniment."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np
import soundfile as sf
from audio_io import resample,write_wav
from download_rvc_assets import sha256
from speech.diagnostics import atomic_json


def extract_stem(source,destination,start,seconds=30,mono=False):
    if not math.isfinite(start) or start < 0 or seconds != 30:
        raise ValueError('Use a finite nonnegative start time and exactly 30 seconds')
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if source == destination or (destination.exists() and os.path.samefile(source,destination)):
        raise ValueError('Prepared stem must not alias the source')
    with sf.SoundFile(str(source)) as audio:
        if audio.channels not in (1,2): raise ValueError('Only mono/stereo stems are supported')
        first=round(start*audio.samplerate); count=round(seconds*audio.samplerate)
        if first+count > len(audio): raise ValueError('Stem does not contain the full requested 30-second interval')
        audio.seek(first); samples=audio.read(count,dtype='float32',always_2d=True)
        sr=audio.samplerate
    if not np.isfinite(samples).all() or np.max(np.abs(samples)) > 1:
        raise ValueError('Nonfinite or out-of-range source stem')
    samples=samples.mean(axis=1) if mono else samples
    samples=resample(samples,sr,48000)
    expected=seconds*48000
    if len(samples) != expected: raise ValueError('Stem resampling did not preserve the interval')
    peak=float(np.max(np.abs(samples)))
    gain=min(1.,.99/peak) if peak else 1.
    write_wav(destination,samples,48000)
    return dict(source_sha256=sha256(source),source_sample_rate=sr,source_start_sample=first,
                source_frames=count,output_frames=expected,channels=1 if mono else samples.shape[1],
                preparation_gain=gain,output_sha256=sha256(destination))


def demonstrate(vocals,instrumental,model,destination,start=0,index=None,index_rate=0,presence_db=0,transpose=0):
    from rvc_adapter import RvcConverter,RvcOptions
    from speech.contracts import SpeechResult
    from export_mix import export_mix
    from voice_autopilot import resolve_ffmpeg_bin
    vocals,instrumental,model=map(lambda p:Path(p).resolve(),(vocals,instrumental,model))
    destination=Path(destination).resolve()
    if destination.exists(): raise ValueError('Choose a new demo directory to preserve earlier listening results')
    if vocals == instrumental or os.path.samefile(vocals,instrumental):
        raise ValueError('Supply separate matching vocal and instrumental stems')
    if any(p == destination or p.is_relative_to(destination) for p in (vocals,instrumental,model)):
        raise ValueError('Demo output must be separate from its inputs')
    # Validate the model before preparing any audio.
    options=RvcOptions(str(model),index=str(Path(index).resolve()) if index else None,index_rate=index_rate,transpose=transpose)
    options.validate()
    ffmpeg=resolve_ffmpeg_bin()
    if not ffmpeg: raise RuntimeError('FFmpeg unavailable')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(destination.parent).free < 10*1024**3+128*1024**2:
        raise RuntimeError('Demo would violate the free-space reserve')
    staging=Path(tempfile.mkdtemp(prefix=destination.name+'.staging-',dir=destination.parent))
    report=dict(status='incomplete',seconds=30,start_seconds=start,model_sha256=sha256(model),
                index_rate=index_rate,index_sha256=sha256(index) if index_rate else None,
                presence_db=presence_db,transpose=transpose,voice_similarity='requires human listening',inputs={})
    atomic_json(staging/'report.json',report)
    try:
        report['inputs']['vocals']=extract_stem(vocals,staging/'source-vocals.wav',start,mono=True)
        report['inputs']['instrumental']=extract_stem(instrumental,staging/'instrumental.wav',start)
        atomic_json(staging/'report.json',report)
        started=time.monotonic()
        RvcConverter(options,diagnostics=staging/'diagnostics').convert(
            SpeechResult(staging/'source-vocals.wav','suno-stem',[]),staging/'converted-vocals.wav')
        report['conversion_seconds']=time.monotonic()-started
        guide,_=sf.read(str(staging/'source-vocals.wav'),dtype='float32')
        converted,_=sf.read(str(staging/'converted-vocals.wav'),dtype='float32')
        source_rms=float(np.sqrt(np.mean(guide.astype(float)**2)))
        converted_rms=float(np.sqrt(np.mean(converted.astype(float)**2)))
        if min(source_rms,converted_rms) <= 1e-5:
            raise ValueError('Vocal stem is silent or converted level is invalid')
        gain=float(np.clip(20*np.log10(source_rms/converted_rms),-12,12))
        report['vocal_mix_gain_db']=gain
        report['gain_method']='One whole-clip RMS match, bounded to +/-12 dB; raw converted stem retained'
        export_mix(staging/'source-vocals.wav',staging/'instrumental.wav',staging/'original-mix.wav',presence_db=0)
        export_mix(staging/'converted-vocals.wav',staging/'instrumental.wav',staging/'your-voice-mix.wav',presence_db=presence_db,vocal_gain_db=gain)
        subprocess.run([ffmpeg,'-nostdin','-v','error','-y','-i',str(staging/'your-voice-mix.wav'),
                        '-ar','48000','-c:a','libmp3lame','-b:a','320k',str(staging/'your-voice-mix.mp3')],check=True,timeout=120)
        metrics={}
        for name in ('source-vocals.wav','instrumental.wav','converted-vocals.wav','original-mix.wav','your-voice-mix.wav'):
            path=staging/name; info=sf.info(str(path)); samples,_=sf.read(str(path),dtype='float32',always_2d=True)
            if (info.samplerate,info.subtype,info.frames) != (48000,'PCM_24',1440000) or not np.isfinite(samples).all():
                raise ValueError('Demo output format/duration/finite contract failed: '+name)
            metrics[name]=dict(sha256=sha256(path),frames=info.frames,channels=info.channels,
                               peak=float(np.max(np.abs(samples))),rms=float(np.sqrt(np.mean(samples.astype(float)**2))))
        for key,path in (('vocals',vocals),('instrumental',instrumental)):
            if sha256(path) != report['inputs'][key]['source_sha256']:
                raise ValueError('Original stem changed during processing')
        report.update(status='complete',outputs=metrics,limiter_peak=.89,final_mp3_sha256=sha256(staging/'your-voice-mix.mp3'))
        atomic_json(staging/'report.json',report)
        staging.rename(destination)
        return destination
    except BaseException as exc:
        report.update(status='failed',error=str(exc))
        atomic_json(staging/'report.json',report)
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vocals',required=True,type=Path)
    parser.add_argument('--instrumental',required=True,type=Path)
    parser.add_argument('--model',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--start',type=float,default=0)
    parser.add_argument('--index',type=Path)
    parser.add_argument('--index-rate',type=float,default=0)
    parser.add_argument('--presence-db',type=float,default=0)
    parser.add_argument('--transpose',type=int,default=0,help='Optional vocal pitch shift in semitones; default preserves the guide melody pitch')
    args=parser.parse_args()
    print(demonstrate(args.vocals,args.instrumental,args.model,args.output_dir,args.start,args.index,args.index_rate,args.presence_db,args.transpose))
