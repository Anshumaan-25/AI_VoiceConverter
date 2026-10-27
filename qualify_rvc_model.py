"""Real pilot model qualification, with retained listening pairs and explicit limits."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import threading
import time

from rvc_training import ROOT,WORKSPACE,verify_cache,child
from speech.diagnostics import atomic_json
from download_rvc_assets import sha256


def check_training_gate(workspace):
    from train_rvc import unresolved_failure
    budget=json.loads((Path(workspace)/'budget.json').read_text())
    if budget.get('active') or unresolved_failure(budget):
        raise RuntimeError('Training safety gate failed or is active; model qualification is blocked')
    if not any(a['stage'] == 'pilot' for a in budget['attempts']):
        raise RuntimeError('A completed bounded pilot is required before qualification')


def build_index(workspace):
    import faiss
    import numpy as np
    cache=verify_cache(workspace)
    index=faiss.IndexFlatL2(768)
    rows=[]
    for clip in cache['clips']:
        if clip['split'] != 'train': continue
        with np.load(child(workspace,clip['file']),allow_pickle=False) as arrays:
            # Training alignment repeats 50 Hz encoder frames; index each original once.
            vectors=np.ascontiguousarray(arrays['features'][::2],dtype=np.float32)
        if index.ntotal+len(vectors) > 200000:
            raise ValueError('Training index exceeds supported vector limit')
        index.add(vectors)
        rows.append(dict(source=clip['source_file'],feature_sha256=clip['sha256'],vectors=len(vectors)))
    target=workspace/'pilot-v2-48k.index'; temporary=target.with_suffix('.tmp')
    faiss.write_index(index,str(temporary))
    if temporary.stat().st_size > 512*1024**2:
        raise ValueError('Training index exceeds supported byte limit')
    from rvc_runtime import load_index
    load_index(temporary,768)
    temporary.replace(target)
    atomic_json(workspace/'index.json',dict(file=target.name,sha256=sha256(target),vectors=index.ntotal,
                feature_manifest_sha256=sha256(workspace/'features.json'),split='train',clips=rows,metric='L2',kind='flat'))


def transcribe(model,path):
    import vosk
    import numpy as np
    from audio_io import read_audio
    audio,sr=read_audio(path,16000)
    recognizer=vosk.KaldiRecognizer(model,sr)
    pcm=(np.clip(audio,-1,1)*32767).astype('<i2').tobytes()
    words=[]
    for offset in range(0,len(pcm),8000):
        if recognizer.AcceptWaveform(pcm[offset:offset+8000]):
            words.append(json.loads(recognizer.Result()).get('text',''))
    words.append(json.loads(recognizer.FinalResult()).get('text',''))
    return ' '.join(words).strip()


def holdout_disagreement(source_asr,converted_asr):
    from benchmark_speech import word_error_rate
    if not source_asr.strip():
        return dict(asr_disagreement=None,asr_score_status='unscorable_empty_source_asr')
    return dict(asr_disagreement=word_error_rate(source_asr,converted_asr)[0],
                asr_score_status='proxy_transcript_comparison_not_ground_truth')


def qualify(workspace,output):
    check_training_gate(workspace)
    import numpy as np
    import soundfile as sf
    import vosk
    from rvc_adapter import RvcConverter,RvcOptions
    from speech.contracts import SpeechOptions,SpeechResult,SpeechError,WorkerTimeout
    from speech.supervisor import IsolatedSpeechGenerator,run_worker
    from rvc_runtime import python_path
    from benchmark_speech import word_error_rate
    from train_rvc import gpu_memory
    model=workspace/'pilot-v2-48k.pth'
    if not model.is_file():
        raise FileNotFoundError('A nonzero-step pilot export is required')
    export=json.loads((workspace/'export.json').read_text())
    if export['step'] < 1 or sha256(model) != export['sha256']:
        raise ValueError('Pilot export identity failed')
    output.mkdir(parents=True,exist_ok=False)
    # CPU-only contained process; the controller never imports the model stack.
    index_dir=output/'index'; index_dir.mkdir()
    run_worker([str(python_path()),'-X','utf8','-B',str(Path(__file__).resolve()),
                '--index-worker',str(index_dir),'--workspace',str(workspace)],
               index_dir,SpeechOptions(device='cpu',load_timeout=120),time.monotonic()+180,backend='cpu')
    vosk.SetLogLevel(-1)
    asr=vosk.Model(str(ROOT/'logs/milestone1/vosk-model-small-en-us-0.15'))
    report=dict(scope='engineering pilot; human voice quality unqualified',model=export,
                index=json.loads((workspace/'index.json').read_text()),records=[],lifecycle=[])
    def flush(): atomic_json(output/'report.json',report)
    flush()
    texts=[
        'Testing the audio pipeline.',
        'Please close the window before you leave the room. We can hear every word clearly, even at the end of the sentence. The small brown dog walked across the quiet garden.',
        'Good morning. Today we are testing the new voice conversion pipeline with a longer recording. Please listen carefully to the beginning and the end of each sentence. We want every word to remain clear as the voice changes. After this test, we will compare the original speech with the converted recording and check the boundaries between the audio segments.'
    ]
    options=RvcOptions(str(model),index=str(workspace/'pilot-v2-48k.index'))
    for number,text in enumerate(texts):
        source=output/f'{number:02d}-source.wav'
        if number == 0:
            import shutil
            shutil.copyfile(ROOT/'logs/milestone1/verified-fallback.wav',source)
        else:
            IsolatedSpeechGenerator().generate(text,source,SpeechOptions(backend='sapi',request_timeout=180))
        for strength in ([0,.5] if number == 0 else [0]):
            target=output/f'{number:02d}-converted-{strength:g}.wav'
            diagnostics=output/f'diagnostics-{number:02d}-{strength:g}'
            before=gpu_memory(); peak=[before or 0]; done=threading.Event()
            def sample():
                while not done.wait(1):
                    value=gpu_memory()
                    if value is not None: peak[0]=max(peak[0],value)
            thread=threading.Thread(target=sample,daemon=True); thread.start()
            started=time.monotonic()
            record=dict(text=text,index_rate=strength,input_sha256=sha256(source),baseline_gpu_mib=before)
            try:
                RvcConverter(replace(options,index_rate=strength),diagnostics=diagnostics).convert(SpeechResult(source,'qualification',[]),target)
                audio,sr=sf.read(str(target),dtype='float32'); info=sf.info(str(target))
                original_info=sf.info(str(source))
                if (sr,info.subtype,info.channels) != (48000,'PCM_24',1) or len(audio) != original_info.frames or not np.isfinite(audio).all():
                    raise ValueError('Converted format/duration contract failed')
                source_text=transcribe(asr,source); converted_text=transcribe(asr,target)
                source_wer=word_error_rate(text,source_text)[0]; converted_wer=word_error_rate(text,converted_text)[0]
                events=[]
                for path in diagnostics.glob('*/events.jsonl'):
                    events.extend(json.loads(line) for line in path.read_text().splitlines())
                joins=[]
                from rvc_audio import segment_ranges
                for start,_,_,_ in list(segment_ranges(len(audio),48000))[1:]:
                    window=audio[max(0,start-480):min(len(audio),start+480)]
                    joins.append(dict(sample=start,max_adjacent_delta=float(np.max(np.abs(np.diff(window))))))
                record.update(status='converted',source_transcript=source_text,converted_transcript=converted_text,
                              source_wer=source_wer,converted_wer=converted_wer,wer_change=converted_wer-source_wer,
                              intelligibility='review' if converted_wer-source_wer > .15 else 'bounded ASR retention',
                              frames=len(audio),peak=float(np.max(np.abs(audio))),rms=float(np.sqrt(np.mean(audio.astype(float)**2))),
                              output_sha256=sha256(target),peak_allocated_bytes=max(e.get('cuda',{}).get('peak_allocated',0) for e in events),
                              peak_reserved_bytes=max(e.get('cuda',{}).get('peak_reserved',0) for e in events),join_windows=joins)
            except Exception as exc:
                record.update(status='failed',error=str(exc))
            finally:
                done.set(); thread.join(timeout=6)
                record.update(elapsed_seconds=time.monotonic()-started,peak_device_gpu_mib=peak[0],after_gpu_mib=gpu_memory())
                record['memory_recovered']=(record['after_gpu_mib'] is not None and before is not None and record['after_gpu_mib'] <= before+64)
                report['records'].append(record); flush()
    # Reconstruction on a target-voice holdout is an additional diagnostic, with
    # source ASR as a proxy transcript rather than a human reference.
    cache=verify_cache(workspace)
    heldout=next(c for c in cache['clips'] if c['split'] == 'val')
    heldout_source=child(Path(cache['dataset']),heldout['source_file'])
    heldout_output=output/'heldout-converted.wav'
    RvcConverter(options,diagnostics=output/'heldout-diagnostics').convert(SpeechResult(heldout_source,'holdout',[]),heldout_output)
    before_text=transcribe(asr,heldout_source); after_text=transcribe(asr,heldout_output)
    report['holdout']=dict(source=heldout['source_file'],source_sha256=sha256(heldout_source),
                           output_sha256=sha256(heldout_output),source_asr=before_text,converted_asr=after_text,
                           **holdout_disagreement(before_text,after_text),
                           limitation='same-session reconstruction; source ASR is not ground truth')
    flush()
    # Stop a worker after actual synthesizer/encoder loading; preserve previous destination bytes.
    source=output/'02-source.wav'
    for mode in ('cancel','timeout'):
        cancel=threading.Event(); finished=threading.Event(); diagnostics=output/('lifecycle-'+mode)
        loaded=threading.Event()
        def watch():
            while not finished.wait(.02):
                for event_file in diagnostics.glob('*/events.jsonl'):
                    if '"stage": "rvc_models_loaded"' in event_file.read_text():
                        loaded.set()
                        if mode == 'cancel': cancel.set()
                        return
        watcher=threading.Thread(target=watch,daemon=True); watcher.start()
        target=output/('preserved-'+mode+'.wav'); target.write_bytes(b'existing destination')
        error=None
        try:
            selected=replace(options,chunk_timeout=.001) if mode == 'timeout' else options
            RvcConverter(selected,cancel=cancel,diagnostics=diagnostics).convert(SpeechResult(source,'qualification',[]),target)
        except (SpeechError,WorkerTimeout) as exc: error=str(exc)
        finally:
            finished.set(); watcher.join(timeout=6)
        report['lifecycle'].append(dict(mode=mode,loaded=loaded.is_set(),error=error,
                        destination_preserved=target.read_bytes() == b'existing destination',gpu_after_mib=gpu_memory()))
        flush()
    recovery=output/'post-cancellation.wav'
    RvcConverter(options,diagnostics=output/'recovery-diagnostics').convert(SpeechResult(output/'00-source.wav','qualification',[]),recovery)
    report['post_cancellation_request']='passed'; flush()
    from gradio_app import build_ui
    from gradio_client import Client
    import socket
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    app=build_ui()
    try:
        app.launch(server_name='127.0.0.1',server_port=port,share=False,prevent_thread_lock=True,quiet=True)
        client=Client(f'http://127.0.0.1:{port}',verbose=False)
        converted=client.predict(str(output/'00-source.wav'),str(model),'','',0,'rmvpe','rvc','',0,0,api_name='/convert_voice')
        generated=client.predict('Testing the audio pipeline.','sapi',False,api_name='/generate_speech')
        audio_path=Path(converted[0]); speech_path=Path(generated[0])
        for path in (audio_path,speech_path):
            info=sf.info(str(path))
            if (info.samplerate,info.subtype,info.channels) != (48000,'PCM_24',1):
                raise ValueError('UI playback/download format mismatch')
        report['ui']=dict(conversion_status=converted[-1],speech_status=generated[-1],audio_contract='48000 PCM_24 mono',
                          converted_sha256=sha256(audio_path),speech_sha256=sha256(speech_path))
    finally:
        app.close()
    flush()
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=WORKSPACE)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'logs/milestone3/qualification')
    parser.add_argument('--index-worker',type=Path)
    args=parser.parse_args()
    if args.index_worker:
        while not (args.index_worker/'go').exists(): time.sleep(.02)
        build_index(args.workspace.resolve())
    else:
        qualify(args.workspace.resolve(),args.output_dir.resolve())
