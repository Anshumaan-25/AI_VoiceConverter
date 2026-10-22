# AI Voice Converter

Windows/Python 3.10 tools for Bark speech generation and so-vits-svc voice conversion. The local GPU target is an RTX 3050 with 4 GB VRAM.

## Current scope

This checkout contains so-vits-svc, Bark/optional Coqui TTS, dataset preparation, a Gradio interface, and CPU stem mixing/export. The Demucs/RVC `pipeline.py` and `test_sweep.py` described in an earlier handoff are **not present**. Their architecture and benchmarks have not been verified here.

See [AUDIT_REPORT.md](AUDIT_REPORT.md) for findings, evidence and remaining qualification work. **TTS is not yet qualified:** actual Bark output contained a carrier tone, and Coqui produced an overlong sample. Guards now reject both failure types. Custom-recording conversion infrastructure is testable, but a trained checkpoint and representative audio validation are still required.

## Setup

```powershell
./setup_autopilot.ps1
./venv310/Scripts/python.exe health_check.py
./venv310/Scripts/python.exe -m unittest discover -s tests -v
```

Setup pins PyTorch/torchaudio 2.1.0 CUDA 12.1 and installs the project runtime. `requirements-lock.txt` records the tested Windows environment. It may require MSVC build tools if a compatible fairseq wheel is not already available. The bootstrap stops on native command failures.

`restore_svc.py` clones upstream revision `730930d337d171479eadf305f96cbed4bb393e77` and applies `patches/so-vits-svc.patch`. It refuses incompatible existing checkouts. This patch preserves the existing local upstream changes as well as the audit repairs. Models, recordings, environments and generated artifacts remain outside Git.

Pretrained weights can be bootstrapped with `./download_pretrain.ps1`. Existing assets are preserved. Only load model files from trusted sources; PyTorch 2.1/pickle checkpoints are executable serialization formats.

## Generate speech

```powershell
./venv310/Scripts/python.exe tts_engine.py --text "Testing the audio pipeline." --model auto --output output.wav
```

Bark uses small models with CPU offloading before model import. Its first run downloads pretrained assets. Text retains sentence punctuation and Unicode; empty generated chunks fail instead of silently dropping words. The default Bark preset is `v2/en_speaker_6`. A speech-only quality guard rejects sustained carrier-like tones. In `auto` mode it releases Bark and uses the CPU Coqui fallback; this changes the source voice and prosody and is logged explicitly. `--model bark` fails on rejected output. Long sustained musical notes can also trigger this guard. Both engines also reject obvious decoder-duration runaways; auto mode can fail if neither result passes. `VOICE_CPU_THREADS` can override the bounded CPU thread setting.

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
