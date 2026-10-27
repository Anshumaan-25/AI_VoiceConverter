# AI Voice Converter

Windows/Python 3.10 tools for speech generation, so-vits-svc and isolated RVC voice conversion. The local GPU target is an RTX 3050 with 4 GB VRAM.

## Current scope

This checkout contains so-vits-svc, an isolated RVC adapter, Bark/optional Coqui TTS, dataset preparation, a Gradio interface, and CPU stem mixing/export. The Demucs/RVC `pipeline.py` and `test_sweep.py` described in an earlier handoff are **not present**. RVC is implemented through `rvc_adapter.py` and `rvc_worker.py`; the earlier handoff's benchmarks do not apply.

See [MILESTONE_1_REPORT.md](MILESTONE_1_REPORT.md) for TTS qualification, [MILESTONE_2_REPORT.md](MILESTONE_2_REPORT.md) for RVC/dataset readiness, [MILESTONE_3_REPORT.md](MILESTONE_3_REPORT.md) for the bounded training pilot and [AUDIT_REPORT.md](AUDIT_REPORT.md) for the earlier audit. RVC runtime/assets and process safety are tested. The pilot stopped on a nonfinite loss after 239 successful paired updates; its last complete checkpoint is step 200. Real trained-voice conversion and acoustic quality remain **unqualified**. Further training is blocked pending numerical diagnosis.

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

`SpeechGenerator`, `AudioValidator`, and `VoiceConverter` protocols live in `speech/contracts.py`. `conversion.SvcConverter` and `rvc_adapter.RvcConverter` implement the conversion boundary for their respective backends, check the persisted audio contract and publish to a separate destination.

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

## RVC runtime and custom dataset

```powershell
./setup_rvc.ps1
./venv-rvc/Scripts/python.exe -B rvc_runtime.py --health
./venv310/Scripts/python.exe -B prepare_rvc_dataset.py --source Anshumaan_RVC.wav
```

RVC uses its own `venv-rvc` environment and a pinned official source checkout under ignored `vendor/rvc`. Setup pins Python 3.10, torch/torchaudio 2.1.0+cu121 and the inference-only dependency lock. A fresh install budgets approximately 8 GiB plus a 10 GiB free-space reserve; actual use depends on existing package caches. Windows fairseq installation may need MSVC build tools when no compatible cached wheel exists. The tested local wheel and isolated imports work on this laptop. Existing TTS packages are not replaced.

`rvc_manifest.json` pins the source commit and the HuBERT/RMVPE assets to a specific upstream revision and LFS SHA256 digests. `restore_rvc.py` refuses unexpected local source changes. `download_rvc_assets.py` stages downloads, checks hashes and preserves mismatched existing assets for investigation. The reviewed upstream patch retains float decoder output until PCM24 storage and makes requested retrieval failures explicit; exact index matches cannot divide by zero.

In the conversion tab select `rvc`, then a **trusted exported RVC inference checkpoint**. Checkpoints use executable PyTorch serialization; a so-vits or training-state checkpoint is a different format. No voice model is downloaded automatically. The feature index is optional at retrieval strength zero. Nonzero strength requires the selected readable L2 index with the model's matching feature width. Single-speaker models use speaker ID zero. Only RMVPE, Harvest and Parselmouth pitch methods are exposed for this backend.

```python
from pathlib import Path
from rvc_adapter import RvcConverter, RvcOptions

# speech is the SpeechResult returned by IsolatedSpeechGenerator.
options = RvcOptions(model=str(Path("models/my_trusted_rvc.pth").resolve()))
converted = RvcConverter(options, diagnostics="logs/rvc").convert(speech, Path("converted.wav"))
```

RVC loads in a contained child interpreter and holds the shared GPU lock through termination. Its explicit CUDA profile is fp16, `x_pad=1`, `x_query=5`, `x_center=30`, `x_max=32`, with the existing allocator ceiling. Five-second core segments retain context and 100 ms crossfades. The default request input limit is ten minutes (programmatically configurable up to one hour); startup/chunk/request deadlines are 180/120/900 seconds. CPU operation can be selected programmatically. These controls are not a substitute for real trained-model memory/quality qualification.

Dataset preparation leaves source files unchanged and writes versioned PCM24/48 kHz clips plus manifests under ignored `datasets/rvc/anshumaan/`. Clips are 2–10 seconds, with VAD context, provenance, exclusions, review flags and exact duplicate checks. Preparation does not denoise or normalize clips. Repeating a completed preparation verifies hashes and reuses its version; interrupted builds are not published. `--source` can be repeated for separate recordings. With one recording, the temporal validation holdout is provisional and has a ten-second exclusion on each side of its boundary. It does not establish independent recording-session quality. The default 60-second usable-duration check is a local safety floor, not proof of enough training data.

The current prepared dataset has 179 training clips (947.26 seconds) and 17 validation clips (97.003 seconds). Its exact source intervals and settings are in the local manifest linked from the milestone report. All 196 clips now have verified, split-isolated content/pitch arrays for the bounded pilot; original WAV files remain unchanged.

### Bounded RVC training pilot

The supported pilot reuses pinned RVC networks/losses in `rvc_training_worker.py`, with float PCM24 loading, explicit frame alignment, batch size 1, AMP, no GPU dataset cache and no data-loader child workers. It does not invoke the upstream DDP/TensorBoard trainer. No additional training packages were required. Optional TensorBoard messages from fairseq remain informational.

For a newly reviewed pilot, the entry points are:

```powershell
./venv310/Scripts/python.exe -B download_rvc_assets.py --training
./venv310/Scripts/python.exe -B prepare_rvc_training.py
./venv310/Scripts/python.exe -B train_rvc.py --stage smoke
./venv310/Scripts/python.exe -B train_rvc.py --stage resume
./venv310/Scripts/python.exe -B train_rvc.py --stage pilot
```

`setup_rvc.ps1 -TrainingAssets` also downloads the hash-verified v2 48 kHz generator/discriminator pair. Training configuration is `configs/rvc_pilot.json`. Across stages the controller permits at most 500 attempted steps and 1800 seconds; a durable reservation prevents a controller crash from resetting the allowance. Actual optimizer updates, skipped updates, losses, sample order and GPU memory are recorded. Complete resume sets contain both networks, optimizer/scheduler/scaler/RNG state and sample position. Interrupted writes never replace the last complete set; two complete sets are retained.

**The current workspace is stopped at a failed safety gate.** Repeating training or `qualify_rvc_model.py` is explicitly rejected. Do not delete/reset its budget or change workspaces to bypass the stop. The existing step-9 inference export predates the failure and is an unqualified diagnostic artifact. Step 200 was verified on CPU after the failure and was not resumed or promoted to an inference export. See the Milestone 3 report for the next numerical investigation and retained evidence.

`qualify_rvc_model.py` supplies the later engineering qualification workflow: a training-only bounded L2 index, repeated real conversions, ASR comparisons, cancellation/deadline checks and queued UI playback/download checks. Its full model workflow remains unexecuted because the training gate failed; it requires the existing optional local Vosk model for ASR. No custom-voice listening pairs were generated in this milestone.

Optional real asset probe, using an existing short speech WAV:

```powershell
./venv310/Scripts/python.exe -B qualify_rvc_runtime.py --audio logs/milestone1/verified-fallback.wav --output-dir logs/rvc-asset-probe-new
```

This probes the real encoder and pitch extractor only. It does not synthesize or qualify a target voice. Existing output directories are refused to preserve earlier evidence.

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
