# AI Voice Converter

Windows/Python 3.10 tools for Bark speech generation and so-vits-svc voice conversion. The local GPU target is an RTX 3050 with 4 GB VRAM.

## Current scope

This checkout contains so-vits-svc, Bark/optional Coqui TTS, dataset preparation, a Gradio interface, and CPU stem mixing/export. The Demucs/RVC `pipeline.py` and `test_sweep.py` described in an earlier handoff are **not present**. Their architecture and benchmarks have not been verified here.

See [MILESTONE_1_REPORT.md](MILESTONE_1_REPORT.md) for current TTS qualification and [AUDIT_REPORT.md](AUDIT_REPORT.md) for the earlier repository audit. English speech now has an isolated Bark path and an offline Windows fallback, with bounded generation and per-chunk validation. Custom voice conversion still requires its own RVC integration, trained checkpoint and representative audio validation.

The completed UI integration and latest verification are recorded in [INTEGRATION_REPORT.md](INTEGRATION_REPORT.md). Start `./venv310/Scripts/python.exe gradio_app.py` and open http://127.0.0.1:7860. Generate speech in the first tab, listen, then use **Use this speech for conversion** to populate the second tab. The UI defaults to the reliable offline Windows voice; the CLI retains its automatic Bark/fallback default. Cancel stops the session's active worker and waits for cleanup. Outputs are retained locally under `logs/app/`; diagnostic retention is optional.

## Setup

```powershell
./setup_autopilot.ps1
./venv310/Scripts/python.exe health_check.py
./venv310/Scripts/python.exe -m unittest discover -s tests -v
```

Setup pins PyTorch/torchaudio 2.1.0 CUDA 12.1 and installs the project runtime. `requirements-lock.txt` records the tested Windows environment. It may require MSVC build tools if a compatible fairseq wheel is not already available. The bootstrap stops on native command failures.

Gradio 3.50.2 also requires the tested web dependency combination: FastAPI 0.161.0 and Starlette 0.52.1. Starlette 1.x removed the template-call compatibility this Gradio version uses. Health checks now exercise HTTP page rendering as well as imports and package consistency.

`restore_svc.py` clones upstream revision `730930d337d171479eadf305f96cbed4bb393e77` and applies `patches/so-vits-svc.patch`. It refuses incompatible existing checkouts. This patch preserves the existing local upstream changes as well as the audit repairs. Models, recordings, environments and generated artifacts remain outside Git.

Pretrained weights can be bootstrapped with `./download_pretrain.ps1`. Existing assets are preserved. Only load model files from trusted sources; PyTorch 2.1/pickle checkpoints are executable serialization formats.

## Generate speech

```powershell
./venv310/Scripts/python.exe tts_engine.py --text "Testing the audio pipeline." --model auto --output output.wav
```

`auto` tries Bark with text/waveform temperatures 0.7/0.7, then the offline Windows SAPI voice if generation fails validation or times out. Coqui remains available through `--model coqui` for diagnostics; its current Tacotron configuration failed one of six benchmark requests and is excluded from automatic mode. Fallback restarts the entire utterance to avoid changing voices between chunks and is logged explicitly.

```powershell
./venv310/Scripts/python.exe tts_engine.py --model sapi --text "Testing the audio pipeline." --output speech.wav
./venv310/Scripts/python.exe tts_engine.py --model bark --text "Testing the audio pipeline." --output speech.wav --diagnostics logs/tts
```

The controller never imports PyTorch. Each backend runs inside a separate process tree with a Windows kill-on-close job. Model loading, each chunk, and the entire request have deadlines (`--load-timeout`, `--chunk-timeout`, `--request-timeout`). Bark bounds semantic decoding; Coqui caps decoder steps and rejects cap hits before vocoder synthesis. Default chunk audio is limited to at most 12 seconds. Timeout or cancellation kills descendants and waits for termination before another backend starts. A failed generation leaves an existing destination unchanged.

Automatic mode reserves part of the whole-request budget for fallback (up to 180 seconds and at most 30% of the requested timeout). This gives the fallback an opportunity to run after a primary timeout; it cannot guarantee completion for every text or unusually short deadline.

Bark retains small models and CPU offloading. CUDA is used when available; `--device cpu` enables explicit CPU diagnostics. When CUDA is unavailable, automatic mode proceeds to SAPI. `VOICE_CPU_THREADS` controls model CPU threads. SAPI uses an installed English Windows voice, preferring Microsoft David Desktop; choose another using `--sapi-voice`. It needs neither network access nor GPU memory, but its voice is less expressive. No input recordings are uploaded. Edge TTS was not selected because its compressed MP3 transport conflicts with the lossless intermediate requirement.

Every chunk is checked for finite samples, bounded duration, silence, persistent tones, saturation and WebRTC speech activity before stitching. Trimming preserves 120 ms padding and gain processing only attenuates. Model-native samples and VAD's 16 kHz PCM16 buffers stay in memory; persisted audio is 48 kHz PCM24. VAD cannot prove that every word was spoken; transcription benchmarks and listening remain separate qualification steps. Singing and multilingual output are not qualified by this milestone.

Diagnostics are optional and saved under a unique directory. They include text/settings, stage events, checkpoint hashes, native sample rates, CUDA/RAM measurements, decoder limits, Bark tokens, Coqui alignments and PCM24 audio. These local traces can contain the input text and generated speech. Failed runs retain their diagnostics; ordinary temporary files are cleaned automatically. First-time model downloads are included in the loading deadline; an explicit model run can use a larger `--load-timeout` and `--request-timeout` if needed.

Programmatic interface:

```python
from pathlib import Path
from speech.contracts import SpeechOptions
from speech.supervisor import IsolatedSpeechGenerator

speech = IsolatedSpeechGenerator().generate(
    "Testing the audio pipeline.", Path("speech.wav"), SpeechOptions())
# A future VoiceConverter adapter consumes speech.path (48 kHz PCM24 WAV).
```

`SpeechGenerator`, `AudioValidator`, and `VoiceConverter` protocols live in `speech/contracts.py`. `conversion.SvcConverter` implements the conversion boundary for the existing so-vits backend, checks the persisted audio contract and publishes to a separate destination. RVC is not implemented by this milestone.

For CPU ASR qualification, install `requirements-validation.txt` and download/extract the English `vosk-model-small-en-us-0.15` model linked in the milestone report. Runtime synthesis does not require Vosk.

```powershell
./venv310/Scripts/python.exe benchmark_speech.py --model sapi --asr-model logs/milestone1/vosk-model-small-en-us-0.15 --output-dir logs/new-benchmark --repeats 2
$env:VOICE_TEST_GPU='1'
./venv310/Scripts/python.exe -m unittest discover -s tests -v
Remove-Item Env:VOICE_TEST_GPU
```

## Prepare custom recordings

```powershell
./venv310/Scripts/python.exe voice_autopilot.py prepare-data --source recordings --speaker anshu
./venv310/Scripts/python.exe voice_autopilot.py preprocess --speech-encoder vec768l12 --f0-predictor rmvpe --skip-loudnorm
./venv310/Scripts/python.exe voice_autopilot.py train --model-name 44k
```

Use independent recordings: processed versions of the same take are not separate validation material. Preparation produces bounded 2–10 second clips, prefers quiet cut points, skips silent clips and records source provenance. This is an energy-based slicer, not a speech recognizer. Exact normalized duplicates are skipped; transformed duplicates still need review. Source recordings are preserved. `--clean-target` archives the old speaker folder instead of deleting it. Preprocessing rebuilds derived audio to avoid stale caches.

Training requires separate recording groups, nonempty splits and at least 20 training clips. This floor prevents accidental tiny-dataset runs; it does not establish sufficient diversity or duration. The two current 93-second recordings and initial `G_0.pth` do not pass readiness checks.

## Convert speech

```powershell
./venv310/Scripts/python.exe gradio_app.py
```

Open http://127.0.0.1:7860 and select a trained checkpoint with its matching configuration. The UI has no default checkpoint. The wrapper rejects `G_0.pth`, missing speakers and unsupported output formats. The CLI supports explicit `--input`, `--model`, `--config`, `--speaker`, `--transpose`, and `--f0-predictor` arguments through `voice_autopilot.py infer`.

The conversion tab accepts WAV/FLAC file uploads without audio-widget re-encoding, preserving PCM24 input bytes. Processing normalizes a copy and leaves the upload unchanged. Conversion inference has a 900-second default process deadline and the same Windows process-tree containment used by speech workers. A qualified trained checkpoint is still required to assess conversion quality.

Model inference is limited to five-second chunks. GPU jobs launched through the wrappers are serialized across processes. A PyTorch allocator cap leaves headroom; it cannot limit other applications' GPU allocations. Optional upstream web servers, diffusion, ONNX exports and realtime modes are not qualified entry points.

## Audio storage and final mix

Persisted intermediates from the supported wrappers use 48 kHz PCM24 WAV. The existing so-vits checkpoint still operates at its configured 44.1 kHz, with 16 kHz encoder features; model-boundary conversions happen in memory. Upsampling does not recover frequencies already lost in a recording.

```powershell
./venv310/Scripts/python.exe export_mix.py --vocals converted.wav --instrumental instrumental.wav --output final.mp3 --presence-db 8
```

The independent CPU exporter applies a one-octave presence EQ centered at approximately 5.657 kHz, mixes both stems, limits peaks and encodes MP3 at an explicit 320 kbps. It requires pre-existing aligned stems; it does not separate songs or perform RVC conversion. Set `--presence-db 0` when no boost is appropriate. Listen to the result: EQ cannot reconstruct missing consonants.

```powershell
./venv310/Scripts/python.exe analyze_baseline.py recording.wav
./venv310/Scripts/python.exe compare_analysis.py baseline.wav candidate.wav
```

Analysis supports PCM16/PCM24/float recordings and reports amplitude, duration, clipping and spectral indicators. Tonal windows are measurements, not automatic proof of a defect.
