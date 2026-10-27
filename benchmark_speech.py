"""Opt-in real synthesis/CPU ASR benchmark; never used to train or modify recordings."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import numpy as np
from audio_io import read_audio
from speech.contracts import SpeechOptions
from speech.supervisor import IsolatedSpeechGenerator

CORPUS = [
    'Testing the audio pipeline.',
    'Hello, my name is David.',
    'Please close the window before you leave the room.',
    'We can hear every word clearly, even at the end of the sentence.',
    'The small brown dog walked across the quiet garden.',
    'Good morning! Are you ready? Let us begin the next part of our project.',
]


def word_error_rate(reference, hypothesis):
    a, b = [re.findall(r"[\w']+", s.lower()) for s in (reference, hypothesis)]
    row = list(range(len(b)+1))
    for i, word in enumerate(a, 1):
        updated = [i]
        for j, other in enumerate(b, 1):
            updated.append(min(updated[-1]+1, row[j]+1, row[j-1]+(word != other)))
        row = updated
    return row[-1]/max(len(a), 1), row[-1], len(a)


def gpu_usage():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=5)
        return [int(line) for line in result.stdout.splitlines()] if result.returncode == 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['auto','bark','coqui','sapi'], default='sapi')
    parser.add_argument('--asr-model', type=Path, required=True, help='Existing Vosk English model directory')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--resume', action='store_true', help='Continue an interrupted benchmark with the same settings')
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error('repeats must be between 1 and 10')
    import vosk
    vosk.SetLogLevel(-1)
    model = vosk.Model(str(args.asr_model))
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    records = []
    report_path = args.output_dir/'report.json'
    if args.resume and report_path.exists():
        previous = json.loads(report_path.read_text(encoding='utf-8'))
        if previous['model'] != args.model or previous['asr_model'] != str(args.asr_model):
            parser.error('Resume settings must match the existing benchmark')
        records = previous['records']
        expected = CORPUS*args.repeats
        if [r['text'] for r in records] != expected[:len(records)] or len(records) > len(expected):
            parser.error('Existing benchmark does not match the requested corpus')
    for repeat in range(args.repeats):
        for index, text in enumerate(CORPUS):
            if repeat*len(CORPUS)+index < len(records):
                continue
            name = f'{repeat:02d}-{index:02d}'
            output = args.output_dir/(name+'.wav')
            before = gpu_usage()
            started = time.monotonic()
            record = dict(text=text, gpu_before_mib=before)
            try:
                result = IsolatedSpeechGenerator().generate(text, output, SpeechOptions(
                    backend=args.model, diagnostics=str(args.output_dir/'diagnostics'),
                    request_timeout=300, chunk_timeout=120))
                audio, sr = read_audio(output, 16000)
                recognizer = vosk.KaldiRecognizer(model, sr)
                transcript = []
                pcm = (np.clip(audio, -1, 1)*32767).astype('<i2').tobytes()
                for offset in range(0, len(pcm), 8000):
                    if recognizer.AcceptWaveform(pcm[offset:offset+8000]):
                        transcript.append(json.loads(recognizer.Result()).get('text',''))
                transcript.append(json.loads(recognizer.FinalResult()).get('text',''))
                transcript = ' '.join(transcript).strip()
                wer, errors, words = word_error_rate(text, transcript)
                record.update(status='passed' if wer <= .35 else 'asr_mismatch', transcript=transcript,
                              wer=wer, word_errors=errors, words=words, backend=result.backend,
                              chunks=result.chunks, path=str(output),
                              sha256=hashlib.sha256(output.read_bytes()).hexdigest())
            except Exception as exc:
                record.update(status='failed', error=str(exc))
            record.update(elapsed_seconds=time.monotonic()-started, gpu_after_mib=gpu_usage())
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            report = dict(model=args.model, asr_model=str(args.asr_model), records=records,
                          aggregate_wer=sum(r.get('word_errors',0) for r in records)/max(1,sum(r.get('words',0) for r in records)),
                          note='ASR provides bounded intelligibility evidence, not a listening-quality guarantee.')
            (args.output_dir/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    aggregate = sum(r.get('word_errors',0) for r in records)/max(1,sum(r.get('words',0) for r in records))
    if any(r['status'] != 'passed' for r in records) or aggregate > .15:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
