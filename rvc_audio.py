"""Sample-counted overlapping segments; no model dependencies."""
import numpy as np
from audio_io import validate_audio


def segment_ranges(frames, sample_rate):
    if frames <= 0 or sample_rate <= 0:
        raise ValueError('Empty audio or invalid sample rate')
    core, overlap, context = 5*sample_rate, sample_rate//10, sample_rate//5
    start = 0
    while start < frames:
        end = min(frames,start+core)
        yield start,end,max(0,start-context),min(frames,end+context)
        if end == frames:
            break
        start = end-overlap


def fit_segment(audio, expected, sample_rate):
    audio = validate_audio(audio,sample_rate)
    if audio.ndim != 1 or abs(len(audio)-expected) > round(.04*sample_rate):
        raise ValueError('RVC segment has unexpected channel count or duration drift')
    if len(audio) < expected:
        audio = np.pad(audio,(0,expected-len(audio)),mode='edge')
    return audio[:expected]


def stitch_segments(parts, sample_rate, expected_frames):
    output = np.zeros(expected_frames,dtype=np.float32)
    weights = np.zeros(expected_frames,dtype=np.float32)
    overlap = sample_rate//10
    for i,(start,audio) in enumerate(parts):
        validate_audio(audio,sample_rate)
        end = start+len(audio)
        if start < 0 or end > expected_frames:
            raise ValueError('Segment lies outside output')
        weight = np.ones(len(audio),dtype=np.float32)
        if i:
            n = min(overlap,len(audio))
            weight[:n] = np.linspace(0,1,n,endpoint=False)
        if i < len(parts)-1:
            n = min(overlap,len(audio))
            weight[-n:] = np.linspace(1,0,n,endpoint=False)
        output[start:end] += audio*weight
        weights[start:end] += weight
    if np.any(weights <= 0):
        raise ValueError('Uncovered samples in RVC segment assembly')
    return output/weights


def validate_conversion(audio, sample_rate, expected_seconds):
    audio = validate_audio(audio,sample_rate)
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio.astype(np.float64)**2)))
    if audio.ndim != 1 or abs(len(audio)/sample_rate-expected_seconds) > .05:
        raise ValueError('Conversion duration or channel count is invalid')
    if rms < 1e-5 or peak > 8:
        raise ValueError('Conversion is silent or has implausible gain')
    return dict(peak=peak,rms=rms,duration=len(audio)/sample_rate)
