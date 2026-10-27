"""Reproducible VAD clips and source-interval splits. Never modifies recordings."""
import argparse
from dataclasses import asdict,dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
import soundfile as sf
from audio_io import resample,validate_audio
from download_rvc_assets import sha256


@dataclass(frozen=True)
class Settings:
    min_seconds: float = 2
    max_seconds: float = 10
    padding_seconds: float = .12
    merge_seconds: float = .3
    guard_seconds: float = 10
    validation_fraction: float = .1
    min_usable_seconds: float = 60
    min_speech_fraction: float = .2
    vad_mode: int = 2

    def validate(self):
        import math
        values = asdict(self)
        if not all(isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v) for v in values.values()):
            raise ValueError('Dataset settings must be finite numbers')
        if not 2 <= self.min_seconds <= self.max_seconds <= 10:
            raise ValueError('Dataset clips must be between 2 and 10 seconds')
        if not 0 <= self.padding_seconds <= .5 or not 0 <= self.merge_seconds <= 1:
            raise ValueError('Invalid speech padding/merge interval')
        if self.guard_seconds < 10 or not .05 <= self.validation_fraction <= .3:
            raise ValueError('Holdout requires a >=10 second guard and 5–30% validation')
        if self.min_usable_seconds < 0 or not 0 < self.min_speech_fraction <= 1:
            raise ValueError('Invalid dataset usability threshold')
        if type(self.vad_mode) is not int or self.vad_mode not in range(4):
            raise ValueError('Invalid WebRTC VAD mode')


def speech_flags(path,mode):
    import webrtcvad
    vad = webrtcvad.Vad(mode)
    flags = []
    with sf.SoundFile(path) as stream:
        info = dict(sample_rate=stream.samplerate,channels=stream.channels,frames=len(stream),subtype=stream.subtype)
        if not len(stream) or stream.channels not in (1,2) or stream.samplerate % 50 or not 8000 <= stream.samplerate <= 96000:
            raise ValueError('Unsupported/empty dataset recording format')
        for block in stream.blocks(blocksize=stream.samplerate,dtype='float32',always_2d=True):
            validate_audio(block,stream.samplerate)
            mono = block.mean(axis=1)
            pcm = (np.clip(resample(mono,stream.samplerate,16000),-1,1)*32767).astype('<i2')
            flags.extend(vad.is_speech(pcm[i:i+320].tobytes(),16000) for i in range(0,len(pcm)-319,320))
    return np.asarray(flags,dtype=bool),info


def regions(flags,sr,left,right,settings):
    """Map VAD frames to exact source samples, merge, then bound all context."""
    hop = sr//50
    active = np.flatnonzero(flags)
    runs = []
    for frame in active:
        a,b = max(left,int(frame)*hop),min(right,(int(frame)+1)*hop)
        if a >= b:
            continue
        if runs and a-runs[-1][1] <= round(settings.merge_seconds*sr):
            runs[-1][1] = b
        else:
            runs.append([a,b])
    padded = []
    for a,b in runs:
        a,b = max(left,a-round(settings.padding_seconds*sr)),min(right,b+round(settings.padding_seconds*sr))
        if padded and a <= padded[-1][1]:
            padded[-1][1] = b
        else:
            padded.append([a,b])
    return padded


def verify_existing(target,sources,identity):
    report = json.loads((target/'manifest.json').read_text(encoding='utf-8'))
    if report['identity'] != identity or report['status'] != 'complete':
        raise ValueError('Existing dataset manifest is incomplete or mismatched')
    for record in report['clips']:
        path = (target/record['file']).resolve()
        if target.resolve() not in path.parents or sha256(path) != record['sha256']:
            raise ValueError('Existing dataset clip is missing, modified or outside its version directory')
    for source in sources:
        if sha256(source['path']) != source['sha256']:
            raise ValueError('Source recording changed')
    return report


def prepare(sources,destination,settings=None):
    settings = settings or Settings()
    settings.validate()
    sources = [Path(p).resolve() for p in sources]
    unique = {}
    for path in sources:
        if not path.is_file() or path.suffix.lower() not in ('.wav','.flac'):
            raise ValueError('Dataset sources must be existing local WAV/FLAC files')
        digest = sha256(path)
        unique.setdefault(digest,dict(path=str(path),name=path.name,sha256=digest))
    if not unique:
        raise ValueError('No recording sources supplied')
    records = [unique[key] for key in sorted(unique)]
    identity = dict(schema=1,source_hashes=sorted(unique),settings=asdict(settings))
    version = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    destination = Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    target = destination/version
    if target.exists():
        return target,verify_existing(target,records,identity)
    if shutil.disk_usage(destination).free < 10*1024**3+sum(Path(r['path']).stat().st_size*2 for r in records):
        raise RuntimeError('Dataset estimate would violate the 10 GiB free-space reserve')
    staged = Path(tempfile.mkdtemp(prefix='.preparing-',dir=destination)).resolve()
    clips,rejected,review = [],[],[]
    seen = set()
    try:
        for source_index,record in enumerate(records):
            flags,info = speech_flags(record['path'],settings.vad_mode)
            record.update(info)
            sr,frames = info['sample_rate'],info['frames']
            if len(records) > 1:
                # Whole source groups; at least one source on either side.
                nval = max(1,round(len(records)*settings.validation_fraction))
                split = 'val' if source_index >= len(records)-nval else 'train'
                intervals = [(split,0,frames)]
            else:
                boundary = int(frames*(1-settings.validation_fraction))//(sr//50)*(sr//50)
                guard = round(settings.guard_seconds*sr)
                intervals = [('train',0,max(0,boundary-guard)),('val',min(frames,boundary+guard),frames)]
                rejected.append(dict(source=record['sha256'],start_sample=max(0,boundary-guard),
                                     end_sample=min(frames,boundary+guard),reason='temporal_split_guard'))
            record['split_intervals'] = [dict(split=s,start_sample=a,end_sample=b) for s,a,b in intervals]
            with sf.SoundFile(record['path']) as stream:
                for split,left,right in intervals:
                    cursor = left
                    for a,b in regions(flags,sr,left,right,settings):
                        if a > cursor:
                            rejected.append(dict(source=record['sha256'],start_sample=cursor,end_sample=a,reason='no_detected_speech'))
                        start = a
                        while start < b:
                            end = min(b,start+round(settings.max_seconds*sr))
                            if end < b:
                                # Choose the quietest 20 ms cut in the last 3 seconds.
                                lo = start+round(max(settings.min_seconds,settings.max_seconds-3)*sr)
                                candidates = range(lo,end,max(1,sr//50))
                                stream.seek(lo)
                                window = stream.read(end-lo+sr//50,dtype='float32',always_2d=True).mean(axis=1)
                                if lo < end:
                                    end = min(candidates,key=lambda cut: float(np.mean(window[cut-lo:cut-lo+sr//50]**2)))
                                if 0 < b-end < round(settings.min_seconds*sr) and b-start <= round(settings.max_seconds*sr):
                                    end = b
                            item = dict(source=record['sha256'],split=split,start_sample=start,end_sample=end)
                            if end-start < round(settings.min_seconds*sr):
                                rejected.append(dict(**item,reason='short_speech_region_or_tail'))
                                start = end
                                continue
                            stream.seek(start)
                            raw = stream.read(end-start,dtype='float32',always_2d=True)
                            validate_audio(raw,sr)
                            audio = resample(raw.mean(axis=1),sr,48000)
                            peak = float(np.max(np.abs(audio)))
                            rms = float(np.sqrt(np.mean(audio.astype(np.float64)**2)))
                            fraction = float(np.mean(flags[start//(sr//50):end//(sr//50)]))
                            if peak > 1 or rms < 1e-5 or not np.isfinite(fraction) or fraction < settings.min_speech_fraction:
                                rejected.append(dict(**item,reason='overshoot_silence_or_low_speech_fraction'))
                                start = end
                                continue
                            pcm = (np.floor(audio.astype(np.float64)*8388608).clip(-8388608,8388607).astype('<i4'))
                            content_hash = hashlib.sha256(pcm.tobytes()).hexdigest()
                            if content_hash in seen:
                                rejected.append(dict(**item,reason='duplicate_pcm'))
                                start = end
                                continue
                            seen.add(content_hash)
                            relative = Path(split)/f"{record['sha256'][:16]}_{start:010d}_{end:010d}.wav"
                            path = staged/relative
                            path.parent.mkdir(parents=True,exist_ok=True)
                            sf.write(path,audio,48000,subtype='PCM_24')
                            duration = len(audio)/48000
                            possible_clip = bool(np.any(np.abs(raw) >= .9999))
                            clip = dict(**item,file=relative.as_posix(),sha256=sha256(path),pcm_sha256=content_hash,
                                        duration_seconds=duration,speech_fraction=fraction,peak=peak,rms=rms,
                                        transforms=['stereo_mean'] if info['channels'] == 2 else [])
                            if sr != 48000:
                                clip['transforms'].append(f'resample_{sr}_to_48000')
                            clips.append(clip)
                            if possible_clip or fraction < .6 or len(review) < 8:
                                review.append(dict(file=clip['file'],source=record['name'],seconds=start/sr,
                                                   possible_clipping=possible_clip,low_speech_fraction=fraction < .6))
                            start = end
                        cursor = b
                    if cursor < right:
                        rejected.append(dict(source=record['sha256'],start_sample=cursor,end_sample=right,reason='no_detected_speech'))
        summary = {s:dict(clips=sum(c['split'] == s for c in clips),
                          seconds=sum(c['duration_seconds'] for c in clips if c['split'] == s)) for s in ('train','val')}
        usable = sum(c['duration_seconds'] for c in clips)
        if not all(summary[s]['clips'] for s in summary) or usable < settings.min_usable_seconds:
            raise ValueError(f'Dataset fails nonempty split/minimum usable duration: {summary}; required {settings.min_usable_seconds}s')
        for record in records:
            if sha256(record['path']) != record['sha256']:
                raise ValueError('Source changed during preparation; refusing publication')
        report = dict(status='complete',identity=identity,sources=records,clips=clips,rejected=rejected,review=review,
                      summary=summary,retained_seconds=usable,
                      duration_range_seconds=[min(c['duration_seconds'] for c in clips),max(c['duration_seconds'] for c in clips)],
                      split_quality='recording_groups' if len(records) > 1 else 'provisional_same_session_temporal_holdout',
                      limitations=['VAD is not speaker identification or transcript validation.',
                                   'Review background noise and speaker consistency before training.'])
        (staged/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        (staged/'review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
        os.rename(staged,target)
        return target,report
    finally:
        if staged.exists():
            # Only remove this function's unique, verified staging directory.
            if staged.parent != destination or not staged.name.startswith('.preparing-'):
                raise RuntimeError('Unsafe staging cleanup path')
            shutil.rmtree(staged)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,action='append',required=True)
    parser.add_argument('--destination',type=Path,default=Path('datasets/rvc/anshumaan'))
    parser.add_argument('--min-usable-seconds',type=float,default=60)
    args = parser.parse_args()
    target,report = prepare(args.source,args.destination,Settings(min_usable_seconds=args.min_usable_seconds))
    print(json.dumps(dict(dataset=str(target),summary=report['summary'],retained_seconds=report['retained_seconds'],
                          split_quality=report['split_quality']),indent=2))


if __name__ == '__main__':
    main()
